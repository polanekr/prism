"""Responsive, user-resizable viewer panels; never modifies image/ROI data."""
from copy import deepcopy
import math
import weakref

from PyQt5.QtCore import QObject, QEvent, QPointF, QTimer, Qt


_DEFAULTS = {
    'detailed': {'raw': True, 'stats': True, 'ratios': {
        'workspace': [.22, .78], 'images': [.35, .65], 'analysis': [.68, .32],
        'profile_horizontal': [.25, .75], 'profile_vertical': [.72, .28]}},
    'compact': {'raw': False, 'stats': False, 'ratios': {
        'workspace': [.23, .77], 'images': [.30, .70], 'analysis': [.72, .28],
        'profile_horizontal': [.17, .83], 'profile_vertical': [.80, .20]}},
}


class ViewerLayout(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window_ref = weakref.ref(window)
        self.preferences = deepcopy(_DEFAULTS)
        self.mode = 'compact' if window.width() < 1600 or window.height() < 900 else 'detailed'
        self.focused = False
        self._applying = False
        self._first_show = True
        self._toolbar_stacked = False
        self.splitters = {
            'workspace': window.workspace_splitter, 'images': window.images_splitter,
            'analysis': window.analysis_splitter, 'profile_horizontal': window.profile_horizontal_splitter,
            'profile_vertical': window.profile_vertical_splitter}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.adapt)
        self._profile_timer = QTimer(self)
        self._profile_timer.setSingleShot(True)
        self._profile_timer.timeout.connect(self.align_horizontal_profile)
        window.viewer_tab.installEventFilter(self)
        window.profile_x_panel.installEventFilter(self)
        window.glw_dose.installEventFilter(self)
        window.plot_dose.getViewBox().sigRangeChanged.connect(self._schedule_profile)
        window.plot_dose.getViewBox().sigResized.connect(self._schedule_profile)
        window.plot_prof_x.getViewBox().sigResized.connect(self._schedule_profile)
        window.img_item_dose.sigImageChanged.connect(self._schedule_profile)
        window.film_boundary.sigRegionChangeFinished.connect(self._schedule_profile)
        for name, splitter in self.splitters.items():
            splitter.setHandleWidth(6)
            splitter.handle(1).setToolTip('Drag to resize the panels.')
            splitter.splitterMoved.connect(lambda pos, index, name=name: self._remember_ratio(name))
        window.combo_view_layout.currentIndexChanged.connect(self.adapt)
        window.btn_map_focus.toggled.connect(self.set_focus)
        window.chk_view_raw.toggled.connect(lambda value: self._set_preference('raw', value))
        window.chk_view_stats.toggled.connect(lambda value: self._set_preference('stats', value))
        self._apply_mode()

    @property
    def window(self):
        reference = getattr(self, '_window_ref', None)
        return reference() if reference is not None else None

    def eventFilter(self, watched, event):
        # Qt can dispatch teardown events after Python has cleared an
        # unreachable controller's attributes during cyclic collection.
        if event.type() not in (QEvent.Resize, QEvent.Show): return False
        window = getattr(self, 'window', None)
        if window is None: return False
        if watched is not window.viewer_tab:
            if event.type() in (QEvent.Resize, QEvent.Show): self._schedule_profile()
            return False
        if event.type() == QEvent.Show and self._first_show:
            self._first_show = False
            QTimer.singleShot(0, self._initialize_visible_layout)
        if event.type() in (QEvent.Resize, QEvent.Show):
            self._timer.start(0)
            self._schedule_profile()
        return False

    def _schedule_profile(self, *_):
        self._profile_timer.start(0)

    def align_horizontal_profile(self):
        """Align the profile's plotting area with the film, excluding axis labels.

        Use logical widget coordinates so screen scaling, map zoom/pan and
        splitter resizing all share the same geometry. No data/ranges change.
        """
        w = self.window
        if w is None: return
        row, canvas = w.profile_x_panel, w.glw_prof_x
        if row.isHidden() or row.width() <= 0: return
        if w.current_dose_map is None:
            canvas.setGeometry(0, 0, row.width(), row.height())
            return
        x1, x2, _, _ = w._film_pixel_bounds()
        view = w.plot_dose.getViewBox()
        def row_x(point):
            point = w.glw_dose.mapFromScene(point)
            return row.mapFromGlobal(w.glw_dose.viewport().mapToGlobal(point)).x()
        left = row_x(view.mapViewToScene(QPointF(x1, 0)))
        right = row_x(view.mapViewToScene(QPointF(x2, 0)))
        map_rect = view.sceneBoundingRect()
        visible_left = row_x(map_rect.topLeft())
        visible_right = row_x(map_rect.bottomRight())
        # When zoomed, keep the complete profile readable within the visible
        # film width instead of allowing its widget to spill beyond the row.
        left, right = max(left, visible_left), min(right, visible_right)
        profile_rect = w.plot_prof_x.getViewBox().sceneBoundingRect()
        margin_left = canvas.mapFromScene(profile_rect.topLeft()).x()
        margin_right = canvas.width()-canvas.mapFromScene(profile_rect.bottomRight()).x()
        margin_left, margin_right = max(0, margin_left), max(0, margin_right)
        left = max(margin_left, left)
        right = min(row.width()-margin_right, right)
        if right <= left:
            left = max(margin_left, visible_left)
            right = min(row.width()-margin_right, visible_right)
        width = max(1, right-left+margin_left+margin_right)
        x = max(0, left-margin_left)
        geometry = (x, 0, min(width, row.width()-x), row.height())
        old = canvas.geometry()
        if (old.x(), old.y(), old.width(), old.height()) != geometry:
            canvas.setGeometry(*geometry)

    def _initialize_visible_layout(self):
        self.adapt()
        self._applying = True
        try:
            if not self.focused: self._resize_splitters()
        finally:
            self._applying = False

    def adapt(self, *_):
        w = self.window
        if w is None: return
        self._wrap_toolbar(w.viewer_tab.width() < 900)
        selection = w.combo_view_layout.currentIndex()
        if selection:
            target = 'detailed' if selection == 1 else 'compact'
        else:
            width, height = w.viewer_tab.width(), w.viewer_tab.height()
            # Hysteresis avoids flicker when a divider is near the threshold.
            target = self.mode
            if width < 1100 or height < 780: target = 'compact'
            elif width >= 1200 and height >= 840: target = 'detailed'
        if target != self.mode:
            self._remember_ratios()
            self.mode = target
            self._apply_mode()

    def _wrap_toolbar(self, stacked):
        if stacked == self._toolbar_stacked: return
        self._toolbar_stacked = stacked
        grid = self.window.viewer_toolbar_grid
        for group in self.window.viewer_tool_groups: grid.removeWidget(group)
        for index, group in enumerate(self.window.viewer_tool_groups):
            grid.addWidget(group, index if stacked else index//2, 0 if stacked else index%2)

    def _remember_ratio(self, name):
        if self._applying or self.focused: return
        w = self.window
        if name == 'images' and w.raw_view_panel.isHidden(): return
        if name == 'analysis' and w.stats_scroll.isHidden(): return
        if name == 'profile_horizontal' and w.glw_prof_y.isHidden(): return
        sizes = self.splitters[name].sizes()
        total = sum(sizes)
        if total:
            self.preferences[self.mode]['ratios'][name] = [value/total for value in sizes]

    def _remember_ratios(self):
        for name in self.splitters: self._remember_ratio(name)

    def _resize_splitters(self, names=None):
        for name in self.splitters if names is None else names:
            ratio = self.preferences[self.mode]['ratios'][name]
            splitter = self.splitters[name]
            extent = splitter.width() if splitter.orientation() == Qt.Horizontal else splitter.height()
            extent = max(1, extent-splitter.handleWidth())
            splitter.setSizes([round(value*extent) for value in ratio])

    def _apply_mode(self):
        self._applying = True
        try:
            w = self.window
            for key, widget in (('raw', w.chk_view_raw), ('stats', w.chk_view_stats)):
                blocked = widget.blockSignals(True)
                widget.setChecked(self.preferences[self.mode][key])
                widget.blockSignals(blocked)
            w.lbl_view_layout.setText('Compact' if self.mode == 'compact' else 'Detailed')
            self.refresh_visibility()
            if not self.focused: self._resize_splitters()
        finally:
            self._applying = False

    def refresh_visibility(self):
        w = self.window
        preferences = self.preferences[self.mode]
        w.raw_view_panel.setVisible(preferences['raw'] and not self.focused)
        w.stats_scroll.setVisible(preferences['stats'] and not self.focused)
        w.glw_prof_y.setVisible(not self.focused and w.combo_prof_mode.currentIndex() == 0)
        w.profile_x_panel.setVisible(not self.focused)
        w.glw_prof_x.setVisible(not self.focused)
        self._schedule_profile()
        w.control_scroll.setVisible(not self.focused)
        w.chk_view_raw.setEnabled(not self.focused)
        w.chk_view_stats.setEnabled(not self.focused)
        w.btn_map_focus.setText('Restore view' if self.focused else 'Enlarge map')

    def _set_preference(self, name, value):
        if self._applying or self.focused: return
        self._remember_ratios()
        self.preferences[self.mode][name] = bool(value)
        splitter_name, pane = ('images', 0) if name == 'raw' else ('analysis', 1)
        if value and self.preferences[self.mode]['ratios'][splitter_name][pane] == 0:
            self.preferences[self.mode]['ratios'][splitter_name] = list(_DEFAULTS[self.mode]['ratios'][splitter_name])
        self._applying = True
        try:
            self.refresh_visibility()
            self._resize_splitters(['images' if name == 'raw' else 'analysis'])
        finally:
            self._applying = False

    def set_focus(self, enabled):
        if enabled == self.focused: return
        self._remember_ratios()
        self.focused = bool(enabled)
        self._applying = True
        try:
            self.refresh_visibility()
            if not self.focused: self._resize_splitters()
        finally:
            self._applying = False
        # No replot, autoRange, setImage or ROI recreation here: the live
        # ViewBox retains its target range, including edits made in focus mode.

    def snapshot(self):
        self._remember_ratios()
        return {'schema_version': 1, 'selection': self.window.combo_view_layout.currentIndex(),
                'preferences': deepcopy(self.preferences)}

    @staticmethod
    def validate(state):
        if not isinstance(state, dict) or type(state.get('schema_version')) is not int or state['schema_version'] != 1:
            raise ValueError('Invalid viewer layout version.')
        if type(state.get('selection')) is not int or not 0 <= state['selection'] <= 2:
            raise ValueError('Invalid viewer layout selection.')
        preferences = state.get('preferences')
        if not isinstance(preferences, dict) or set(preferences) != set(_DEFAULTS):
            raise ValueError('Invalid viewer layout modes.')
        for values in preferences.values():
            if not isinstance(values, dict) or set(values) != {'raw', 'stats', 'ratios'}:
                raise ValueError('Invalid viewer layout preferences.')
            if type(values['raw']) is not bool or type(values['stats']) is not bool:
                raise ValueError('Invalid viewer visibility.')
            ratios = values['ratios']
            if not isinstance(ratios, dict) or set(ratios) != set(_DEFAULTS['compact']['ratios']):
                raise ValueError('Invalid viewer splitters.')
            for ratio in ratios.values():
                if (not isinstance(ratio, list) or len(ratio) != 2
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               or not 0 <= value <= 1 for value in ratio)
                        or not math.isclose(sum(ratio), 1., abs_tol=1e-6)):
                    raise ValueError('Invalid viewer panel proportions.')

    def restore(self, state):
        self.validate(state)
        self.preferences = deepcopy(state['preferences'])
        w = self.window
        blocked = w.combo_view_layout.blockSignals(True)
        w.combo_view_layout.setCurrentIndex(state['selection'])
        w.combo_view_layout.blockSignals(blocked)
        blocked = w.btn_map_focus.blockSignals(True)
        w.btn_map_focus.setChecked(False)
        w.btn_map_focus.blockSignals(blocked)
        self.focused = False
        selection = state['selection']
        self.mode = ('detailed' if selection == 1 else 'compact') if selection else self.mode
        self._apply_mode()
        self.adapt()
