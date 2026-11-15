"""Experimental data sheet and native-resolution film-plane analysis UI."""
from pathlib import Path
from copy import deepcopy
import json
import html
import numpy as np
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLineEdit,QComboBox,QPushButton,
                            QLabel,QScrollArea,QSplitter,QTabWidget,QTextBrowser,QFileDialog,QSizePolicy)
import pyqtgraph as pg
from beam_analysis import Experiment, analyze_measured_footprint
from bio_analysis import SampleRegion
from background_tasks import TaskController
from project_state import atomic_json, file_record


class BeamAnalysisPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window=window; self.records={}; self.bound_film=None; self.result=None; self.loading=False
        self.tasks=TaskController(self)
        self.tasks.ready.connect(self.display)
        self.tasks.failed.connect(lambda message:self.status.setText(message))
        root=QVBoxLayout(self)
        note=QLabel('Measured FILM-plane analysis. No source reconstruction, jitter separation or biological FLASH classification. '
                    'Shot counts are user supplied; blank values remain unknown.')
        note.setWordWrap(True);root.addWidget(note)
        actions=QHBoxLayout();root.addLayout(actions)
        split=QSplitter();root.addWidget(split)
        form_widget=QWidget();form=QFormLayout(form_widget);form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.inputs={}
        labels={'exposure_id':'Exposure ID','shots':'Counted irradiation events N','duration_s':'Exposure / gate duration [s]',
            'start_s':'Alternative window start [s]','end_s':'Alternative window end [s]',
            'frequency_hz':'Nominal repetition frequency [Hz]','particle_pulse_fs':'Particle pulse duration [fs]',
            'laser_pulse_fs':'Laser pulse FWHM [fs] (metadata only)','background_gy':'Background for shape weights [Gy]',
            'core_radius_mm':'Core radius for halo [mm] (optional)'}
        for key,label in labels.items():
            widget=QLineEdit('0' if key=='background_gy' else '')
            widget.setPlaceholderText('Unknown' if key not in ('exposure_id','background_gy') else '')
            widget.textChanged.connect(self.edited);self.inputs[key]=widget;form.addRow(label,widget)
        self.choices={}
        for key,label,items in (
            ('count_basis','Count provenance',[('Reported / counted','reported'),('Estimated','estimated')]),
            ('pulse_plane','Particle duration applies at',[('Unknown','unknown'),('Film plane','film'),('Source (not film)','source')]),
            ('pulse_structure','Particles per counted event',[('Unknown / multiple pulses','unknown'),('One particle pulse per event (declared)','one_per_event')]),
            ('pulse_duration_basis','Particle duration provenance',[('Assumed / model scenario','assumed'),('Reported measurement','reported')]),
            ('pulse_shape','Particle pulse convention',[('Unknown','unknown'),('Rectangular duration','rectangular'),('Gaussian FWHM','gaussian_fwhm')])):
            combo=QComboBox()
            for text,value in items:combo.addItem(text,value)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);combo.setMinimumContentsLength(10)
            combo.currentIndexChanged.connect(self.edited);self.choices[key]=combo;form.addRow(label,combo)
        self.region_mode=QComboBox();self.region_mode.addItems(['Full selected film rectangle','Beam Analysis ellipse'])
        self.region_mode.currentIndexChanged.connect(self.invalidate);form.addRow('Analysis support',self.region_mode)
        self.run_button=QPushButton('Calculate');self.run_button.clicked.connect(self.run)
        actions.addWidget(self.run_button)
        self.cancel_button=QPushButton('Cancel');self.cancel_button.clicked.connect(self.invalidate);actions.addWidget(self.cancel_button)
        self.export_button=QPushButton('Export JSON + NPZ');self.export_button.clicked.connect(self.export)
        actions.addWidget(self.export_button)
        actions.addStretch()
        for widget in (*self.inputs.values(),*self.choices.values(),self.region_mode):
            form.labelForField(widget).setWordWrap(True)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(form_widget);scroll.setMinimumWidth(260);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);split.addWidget(scroll)
        views=QTabWidget();views.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored);split.addWidget(views);split.setSizes([350,850])
        map_page=QWidget();map_layout=QVBoxLayout(map_page)
        self.map_choice=QComboBox();self.map_choice.addItems(['Dose [Gy]','Mean dose / event [Gy]','Exposure-average rate [Gy/s]','Conditional model peak rate [Gy/s]'])
        self.map_choice.currentIndexChanged.connect(self.draw_map);map_layout.addWidget(self.map_choice)
        self.canvas=pg.GraphicsLayoutWidget();self.plot=self.canvas.addPlot();self.plot.setAspectLocked(True);self.plot.invertY(True)
        self.plot.setLabel('bottom','Film X [mm]');self.plot.setLabel('left','Film Y [mm]')
        self.image=pg.ImageItem();self.plot.addItem(self.image)
        self.lut=pg.HistogramLUTItem();self.lut.setImageItem(self.image);self.lut.gradient.loadPreset('viridis');self.canvas.addItem(self.lut)
        self.center=pg.ScatterPlotItem(symbol='+',size=15,pen='r');self.plot.addItem(self.center)
        self.circles=[];map_layout.addWidget(self.canvas);views.addTab(map_page,'Film map')
        self.enclosed=pg.PlotWidget(title='Enclosed background-subtracted dose integral (ROI only)')
        self.enclosed.setLabel('bottom','Radius from centroid [mm]');self.enclosed.setLabel('left','Enclosed signal [%]')
        self.curve=self.enclosed.plot(pen='c');views.addTab(self.enclosed,'Enclosed signal')
        self.report=QTextBrowser();views.addTab(self.report,'Metrics & assumptions')
        self.status=QLabel('Load a dose map, complete available fields, then calculate.');self.status.setWordWrap(True);root.addWidget(self.status)
        self.export_button.setEnabled(False)

    def read_experiment(self):
        values={}
        for key,widget in self.inputs.items():
            text=widget.text().strip()
            if key=='exposure_id':values[key]=text
            elif key=='shots':values[key]=int(text) if text else None
            else:values[key]=float(text) if text else (0. if key=='background_gy' else None)
        values.update({key:combo.currentData() for key,combo in self.choices.items()})
        return Experiment(**values)

    def edited(self,*_):
        if self.loading:return
        self.invalidate()
        try:
            e=self.read_experiment()
            if self.bound_film:self.records[self.bound_film]=e.to_dict()
            self.status.setText('Inputs changed. Recalculate. Values apply to this film exposure only.')
        except (ValueError,OverflowError) as error:self.status.setText(str(error))

    def bind_film(self):
        name=getattr(self.window,'current_film_name',None) if self.window.current_dose_map is not None else None
        if name==self.bound_film:return
        self.bound_film=name;self.loading=True
        try:
            e=Experiment(**self.records.get(name,{}))
            for key,widget in self.inputs.items():
                value=getattr(e,key);widget.setText('' if value is None else str(value))
            for key,combo in self.choices.items():combo.setCurrentIndex(combo.findData(getattr(e,key)))
        finally:self.loading=False
        self.invalidate()
        self.status.setText('Exposure data bound to film: '+str(name))

    def snapshot(self):
        if self.bound_film:self.records[self.bound_film]=self.read_experiment().to_dict()
        return {'schema_version':1,'records':deepcopy(self.records),'region_mode':self.region_mode.currentIndex()}

    @staticmethod
    def validate_state(state):
        if state.get('schema_version')!=1 or type(state.get('region_mode')) is not int or state['region_mode'] not in (0,1):
            raise ValueError('Invalid beam analysis state.')
        if not isinstance(state.get('records'),dict):raise ValueError('Invalid exposure records.')
        for key,value in state['records'].items():
            if not isinstance(key,str) or not key:raise ValueError('Invalid film exposure key.')
            Experiment(**value)

    def restore(self,state):
        self.validate_state(state);self.records=deepcopy(state['records']);self.bound_film=None
        self.region_mode.setCurrentIndex(state['region_mode']);self.bind_film()

    def invalidate(self,*_):
        self.tasks.cancel();self.result=None;self.export_button.setEnabled(False)
        self.status.setText("Analysis invalidated. Calculate to refresh results.")
        self.image.clear();self.center.clear();self.curve.clear();self.report.clear()
        for circle in self.circles:self.plot.removeItem(circle)
        self.circles=[]

    def run(self):
        self.bind_film();self.invalidate();w=self.window
        if w.current_dose_map is None:self.status.setText('Load a dose map first.');return
        try:
            experiment=self.read_experiment();self.records[self.bound_film]=experiment.to_dict()
            bounds=w._film_pixel_bounds();x1,x2,y1,y2=bounds
            ellipse=self.region_mode.currentIndex()==1
            if ellipse:
                shape,vectors,origin=w.roi.getAffineSliceParams(w.current_dose_map,w.img_item_dose)
                region=SampleRegion(0,tuple(shape),tuple(tuple(v) for v in vectors),tuple(origin),None,None)
            else:region=SampleRegion(0,(y2-y1,x2-x1),((1,0),(0,1)),(y1,x1),None,None)
            spacing=float(w.current_meta.get('pixel_spacing_mm',.0846))
            dose=np.array(w.current_dose_map,copy=True);sigma=np.array(w.current_std_map,copy=True) if w.current_std_map is not None else None
            provenance={'film':str(getattr(w,'current_dose_path','')),'film_name':self.bound_film,'film_bounds_px':list(bounds),
                'region_mode':'ellipse' if ellipse else 'film_rectangle','roi_shape':list(region.shape),
                'roi_vectors':[list(v) for v in region.vectors],'roi_origin':list(region.origin),
                'processing':deepcopy(w.current_meta),'spacing_mm':spacing}
            def calculate(check,progress):
                import hashlib
                provenance['dose_array_sha256']=hashlib.sha256(dose.tobytes()).hexdigest()
                if sigma is not None:provenance['sigma_array_sha256']=hashlib.sha256(sigma.tobytes()).hexdigest()
                result=analyze_measured_footprint(dose,sigma,spacing,region,bounds,experiment,check,ellipse=ellipse)
                result['metrics']['provenance']=provenance
                return result
            self.status.setText('Calculating native-resolution film metrics…');self.tasks.submit(calculate)
        except (ValueError,TypeError,OverflowError) as error:self.status.setText(str(error))

    def display(self,result):
        self.result=result;self.export_button.setEnabled(True)
        m=result['metrics'];e=Experiment(**m['experiment'])
        available=(True,e.shots is not None,e.exposure_seconds is not None,m['temporal']['conditional_peak_rate_gy_s'] is not None)
        blocked=self.map_choice.blockSignals(True)
        for i,enabled in enumerate(available):self.map_choice.model().item(i).setEnabled(enabled)
        if not available[self.map_choice.currentIndex()]:self.map_choice.setCurrentIndex(0)
        self.map_choice.blockSignals(blocked);self.draw_map()
        m=result['metrics'];self.curve.setData(result['radii'],result['fractions']*100)
        def fmt(value):return 'N/A' if value is None else f'{value:.5g}'
        t=m['temporal'];r=m['radii_mm'] or {}
        rows=[('ROI area [mm²]',m['roi_area_mm2']),('Mean dose [Gy]',m['mean_dose_gy']),
            ('Minimum / maximum dose [Gy]',f"{fmt(m['min_dose_gy'])} / {fmt(m['max_dose_gy'])}"),
            ('Spatial dose SD [Gy]',m['spatial_std_gy']),('Mean pixel sigma bound [Gy] (not an independent-pixel SEM)',m['mean_pixel_sigma_bound_gy']),
            ('Dose integral [Gy mm²] (not deposited energy)',m['dose_integral_gy_mm2']),
            ('Background for shape weights [Gy]',e.background_gy),
            ('Centroid X / Y [mm]', 'N/A' if m['centroid_mm'] is None else ' / '.join(fmt(v) for v in m['centroid_mm'])),
            ('Major / minor RMS width [mm]',f"{fmt(m['sigma_major_mm'])} / {fmt(m['sigma_minor_mm'])}"),
            ('Ellipticity (major/minor)',m['ellipticity_major_minor']),('Major-axis angle [deg, Y down; N/A for circular spot]',m['orientation_deg_y_down']),
            ('R50 / R80 / R90 [mm] (fraction of retained signal)', ' / '.join(fmt(r.get(str(p))) for p in (50,80,90))),
            ('Halo signal outside declared core [%]',m['halo_signal_fraction']*100 if m['halo_signal_fraction'] is not None else None),
            ('Halo / core signal ratio (ROI only)',m['halo_core_ratio']),('Declared core radius [mm]',e.core_radius_mm),
            ('Events N',e.shots),('Count basis',e.count_basis),('Exposure window [s]',e.exposure_seconds),
            ('Mean dose per counted event [Gy]',t['dose_per_shot_gy']),('Exposure-average dose rate [Gy/s]',t['average_rate_gy_s']),
            ('Conditional model PEAK rate [Gy/s]',t['conditional_peak_rate_gy_s']),
            ('Particle duration [fs] / plane / convention',f"{fmt(e.particle_pulse_fs)} / {e.pulse_plane} / {e.pulse_shape}"),
            ('Particle duration provenance / pulse structure',e.pulse_duration_basis+' / '+e.pulse_structure),
            ('Laser pulse [fs] (metadata only)',e.laser_pulse_fs)]
        table=''.join('<tr><td>'+html.escape(label)+'</td><td>'+html.escape(fmt(value) if isinstance(value,(int,float)) or value is None else str(value))+'</td></tr>' for label,value in rows)
        self.report.setHtml('<h3>Measured film-plane footprint</h3><table cellpadding="5">'+table+'</table><p>'+
            html.escape(t['pulse_rate_note'])+'</p><p>'+html.escape(t['uncertainty_note']+' '+m['shape_uncertainty_note'])+'</p><p>'+html.escape(' '.join(m['warnings']))+'</p>')
        self.status.setText(f"ROI mean {fmt(m['mean_dose_gy'])} Gy | Mean/event {fmt(t['dose_per_shot_gy'])} Gy | "
            f"Exposure average {fmt(t['average_rate_gy_s'])} Gy/s | Conditional model peak {fmt(t['conditional_peak_rate_gy_s'])} Gy/s\n"
            +t['pulse_rate_note']+'\n'+' '.join(m['warnings']))

    def draw_map(self,*_):
        if self.result is None:return
        r=self.result;m=r['metrics'];t=m['temporal'];index=self.map_choice.currentIndex()
        e=Experiment(**m['experiment']);factor=1.
        if index==1:factor=1/e.shots if e.shots is not None else None
        elif index==2:factor=1/e.exposure_seconds if e.exposure_seconds is not None else None
        elif index==3:
            rate=t['conditional_peak_rate_gy_s']
            factor=rate/m['mean_dose_gy'] if rate is not None and m['mean_dose_gy']>0 else (0. if rate==0 else None)
        if factor is None:
            self.image.clear();self.center.clear()
            for circle in self.circles:self.plot.removeItem(circle)
            self.circles=[];self.status.setText('Selected map unavailable: '+t['pulse_rate_note'] if index==3 else 'Selected map requires the corresponding exposure input.')
            return
        values=np.where(r['mask'],r['dose_crop']*factor,np.nan)
        finite=values[np.isfinite(values)]
        lo,hi=np.percentile(finite,[1,99.5])
        if hi<=lo:hi=lo+max(abs(float(lo))*.05,1e-4)
        self.image.setImage(values,autoLevels=False,levels=(lo,hi))
        self.lut.setLevels(lo,hi);self.lut.setHistogramRange(lo,hi,padding=.06)
        x,y=r['origin_mm'];h,w=values.shape;s=r['spacing_mm'];self.image.setRect(QRectF(x-.5*s,y-.5*s,w*s,h*s))
        self.plot.setTitle(self.map_choice.currentText()+' — film plane')
        for circle in self.circles:self.plot.removeItem(circle)
        self.circles=[]
        if m['centroid_mm'] is not None:
            cx,cy=m['centroid_mm'];self.center.setData([cx],[cy])
            theta=np.linspace(0,2*np.pi,181)
            for p,color in ((50,'y'),(80,'c'),(90,'m')):
                radius=m['radii_mm'][str(p)]
                curve=self.plot.plot(cx+radius*np.cos(theta),cy+radius*np.sin(theta),pen=pg.mkPen(color,style=Qt.DashLine))
                self.circles.append(curve)
            if e.core_radius_mm is not None:
                radius=e.core_radius_mm
                self.circles.append(self.plot.plot(cx+radius*np.cos(theta),cy+radius*np.sin(theta),pen=pg.mkPen('w')))
            cov=np.asarray(m['covariance_mm2']);eigen,axes=np.linalg.eigh(cov)
            for i in (0,1):
                delta=2*np.sqrt(max(0.,eigen[i]))*axes[:,i]
                self.circles.append(self.plot.plot([cx-delta[0],cx+delta[0]],[cy-delta[1],cy+delta[1]],pen=pg.mkPen('r')))
        self.plot.autoRange()

    def export(self):
        if self.result is None:return
        name,_=QFileDialog.getSaveFileName(self,'Export measured beam analysis',str(self.bound_film or 'beam')+'_beam.json','JSON (*.json)')
        if not name:return
        try:
            path=Path(name);metrics=deepcopy(self.result['metrics'])
            source=metrics['provenance']['film']
            if source:metrics['provenance']['film_file_at_export']=file_record(source)
            extra={'sigma_crop':self.result['sigma_crop']} if self.result['sigma_crop'] is not None else {}
            np.savez_compressed(path.with_suffix('.npz'),**extra,dose_crop=self.result['dose_crop'],mask=self.result['mask'],
                spacing_mm=self.result['spacing_mm'],origin_mm=self.result['origin_mm'],
                radii=self.result['radii'],enclosed_fraction=self.result['fractions'])
            atomic_json(path,metrics);self.status.setText('Saved JSON metrics and NPZ measured arrays.')
        except (OSError,ValueError) as error:self.status.setText(str(error))
