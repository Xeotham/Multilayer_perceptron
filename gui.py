import sys
import os
import pandas as pd
import numpy as np

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QTabWidget, QCheckBox, QComboBox,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QFormLayout, QHeaderView,
                             QSplitter, QScrollArea, QGraphicsView,
                             QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem,
                             QGraphicsTextItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPainter

# --- Matplotlib Imports ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Library Imports ---
from ML_MHAOUAS import MLP, Layers, train_test_split, save_in_file, load_from_file
from ML_MHAOUAS.Preprocessing import LabelBinarizer, StandardScaler
from ML_MHAOUAS.FeatureSelection import VarianceThreshold
from ML_MHAOUAS.Pipeline import make_pipeline


# ------------------------------------------------------------------------
# CUSTOM WIDGET: MATPLOTLIB CANVAS
# ------------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_train = fig.add_subplot(121)
        self.axes_val = fig.add_subplot(122)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


# ------------------------------------------------------------------------
# POPUP WINDOW: PLOTS
# ------------------------------------------------------------------------
class PlotWindow(QWidget):
    def __init__(self, mlp_ref):
        super().__init__()
        self.setWindowTitle("Training Metrics - Real-time")
        self.setGeometry(150, 150, 900, 500)
        self.mlp = mlp_ref

        layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=10, height=5, dpi=100)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Timer to auto-refresh plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_charts)
        self.timer.start(500)  # Refresh every 500ms
        self.update_charts()

    def update_charts(self):
        if not self.mlp: return

        self.canvas.axes_train.cla()
        self.canvas.axes_val.cla()

        # Plot Training Cost
        if hasattr(self.mlp, 'cost_evolution') and self.mlp.cost_evolution:
            self.canvas.axes_train.plot(self.mlp.cost_evolution, 'r-', label='Train Cost')
            self.canvas.axes_train.set_title("Training Cost")
            self.canvas.axes_train.grid(True)
            self.canvas.axes_train.legend()

        # Plot Validation Cost
        if hasattr(self.mlp, 'val_cost_evolution') and self.mlp.val_cost_evolution:
            self.canvas.axes_val.plot(self.mlp.val_cost_evolution, 'b-', label='Val Cost')
            self.canvas.axes_val.set_title("Validation Cost")
            self.canvas.axes_val.grid(True)
            self.canvas.axes_val.legend()

        self.canvas.draw()


# ------------------------------------------------------------------------
# CUSTOM WIDGET: ANIMATED NETWORK VISUALIZER
# ------------------------------------------------------------------------
class NetworkVisualizer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.setStyleSheet("border: 1px solid #444;")

        # Structure State
        self.input_size = 0
        self.output_size = 0
        self.hidden_layers_config = []

        # Animation State
        self.layers_visuals = []
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate_step)
        self.current_anim_layer = 0
        self.is_animating = False

    def set_structure(self, input_size, hidden_configs, output_size):
        self.input_size = input_size
        self.hidden_layers_config = hidden_configs
        self.output_size = output_size
        self.redraw()

    def start_animation(self):
        if not self.layers_visuals: return
        self.is_animating = True
        self.current_anim_layer = 0
        self.animation_timer.start(150)

    def stop_animation(self):
        self.is_animating = False
        self.animation_timer.stop()
        self.reset_styles()

    def _animate_step(self):
        if not self.layers_visuals: return
        self.reset_styles()

        idx = self.current_anim_layer
        layer_data = self.layers_visuals[idx]

        # Highlight Nodes
        highlight_brush = QBrush(QColor(255, 255, 255))
        for node in layer_data['nodes']:
            if node: node.setBrush(highlight_brush)

        # Highlight Lines
        highlight_pen = QPen(QColor(0, 255, 255))
        highlight_pen.setWidth(2)
        for line in layer_data['lines']:
            if line:
                line.setPen(highlight_pen)
                line.setZValue(1)

        self.current_anim_layer += 1
        if self.current_anim_layer >= len(self.layers_visuals):
            self.current_anim_layer = 0

    def reset_styles(self):
        default_line_pen = QPen(QColor(255, 255, 255, 30))
        default_line_pen.setWidth(1)

        for layer in self.layers_visuals:
            for node in layer['nodes']:
                if node: node.setBrush(QBrush(layer['color']))
            for line in layer['lines']:
                if line:
                    line.setPen(default_line_pen)
                    line.setZValue(-1)

    def redraw(self):
        self.stop_animation()
        self.scene.clear()
        self.layers_visuals = []

        layers = []
        layers.append({'count': self.input_size, 'type': 'input', 'name': 'Input'})
        for cfg in self.hidden_layers_config:
            layers.append({'count': cfg['count'], 'type': 'hidden', 'name': cfg['name']})
        layers.append({'count': self.output_size, 'type': 'output', 'name': 'Output'})

        if not layers: return

        view_width = self.width() if self.width() > 100 else 800
        view_height = self.height() if self.height() > 100 else 600
        margin_x = 50
        usable_w = view_width - 2 * margin_x
        layer_spacing = usable_w / (len(layers) - 1) if len(layers) > 1 else usable_w
        max_visual = 16
        radius = 10

        all_layer_coords = []

        for i, layer in enumerate(layers):
            x = margin_x + i * layer_spacing
            count = layer['count']

            if layer['type'] == 'input':
                color = QColor("#3498db")
            elif layer['type'] == 'output':
                color = QColor("#e74c3c")
            else:
                color = QColor("#2ecc71")

            visual_data = {'nodes': [], 'lines': [], 'color': color}

            to_draw = min(count, max_visual)
            is_compressed = count > max_visual
            total_h = to_draw * (radius * 3)
            start_y = (view_height - total_h) / 2

            curr_coords = []

            # Label
            txt = QGraphicsTextItem(f"{layer['name']}\n({count})")
            txt.setDefaultTextColor(QColor("white"))
            txt.setFont(QFont("Arial", 8))
            txt.setPos(x - 20, 20)
            self.scene.addItem(txt)

            for j in range(to_draw):
                if is_compressed and j == to_draw // 2:
                    dots = QGraphicsTextItem("...")
                    dots.setDefaultTextColor(QColor("white"))
                    dots.setPos(x - 5, start_y + j * (radius * 3))
                    self.scene.addItem(dots)
                    curr_coords.append(None)
                    visual_data['nodes'].append(None)
                    continue

                y = start_y + j * (radius * 3)
                ellipse = QGraphicsEllipseItem(x, y, radius * 2, radius * 2)
                ellipse.setBrush(QBrush(color))
                ellipse.setPen(QPen(Qt.GlobalColor.white))
                self.scene.addItem(ellipse)

                visual_data['nodes'].append(ellipse)
                curr_coords.append(QPointF(x + radius, y + radius))

            all_layer_coords.append(curr_coords)
            self.layers_visuals.append(visual_data)

        # Draw Connections
        for i in range(len(all_layer_coords) - 1):
            curr = all_layer_coords[i]
            nxt = all_layer_coords[i + 1]
            src_visuals = self.layers_visuals[i]

            for s_pt in curr:
                if s_pt is None: continue
                for e_pt in nxt:
                    if e_pt is None: continue
                    line = QGraphicsLineItem(s_pt.x(), s_pt.y(), e_pt.x(), e_pt.y())
                    line.setPen(QPen(QColor(255, 255, 255, 30)))
                    line.setZValue(-1)
                    self.scene.addItem(line)
                    src_visuals['lines'].append(line)

    def resizeEvent(self, event):
        was_animating = self.is_animating
        self.redraw()
        if was_animating: self.start_animation()
        super().resizeEvent(event)


# ------------------------------------------------------------------------
# WIDGET: LAYER ROW
# ------------------------------------------------------------------------
class LayerConfigRow(QWidget):
    removed = pyqtSignal(QWidget)
    changed = pyqtSignal()

    def __init__(self, index):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)

        self.index_lbl = QLabel(f"L{index}:")
        self.index_lbl.setFixedWidth(25)

        self.name_edit = QLineEdit(f"Hidden {index}")
        self.name_edit.setPlaceholderText("Name")
        self.name_edit.textChanged.connect(self.emit_changed)

        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(1, 10000)
        self.neurons_spin.setValue(30)
        self.neurons_spin.valueChanged.connect(self.emit_changed)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "sigmoid", "softmax"])
        self.activation_combo.currentIndexChanged.connect(self.emit_changed)

        self.btn_remove = QPushButton("x")
        self.btn_remove.setFixedWidth(25)
        self.btn_remove.setStyleSheet("background-color: #e74c3c; color: white;")
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))

        layout.addWidget(self.index_lbl)
        layout.addWidget(self.name_edit)
        layout.addWidget(QLabel("N:"))
        layout.addWidget(self.neurons_spin)
        layout.addWidget(self.activation_combo)
        layout.addWidget(self.btn_remove)
        self.setLayout(layout)

    def emit_changed(self): self.changed.emit()

    def get_config(self):
        return {'count': self.neurons_spin.value(),
                'activation': self.activation_combo.currentText(),
                'name': self.name_edit.text()}


# ------------------------------------------------------------------------
# WORKER THREAD
# ------------------------------------------------------------------------
class TrainingWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pipeline, X_train, y_train):
        super().__init__()
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train

    def run(self):
        try:
            self.pipeline.fit(self.X_train, self.y_train)
            self.finished.emit(self.pipeline)
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())


# ------------------------------------------------------------------------
# MAIN GUI
# ------------------------------------------------------------------------
class MLPGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML_MHAOUAS GUI Manager")
        self.setGeometry(100, 100, 1400, 900)
        self.current_mlp = None  # Store reference for plotting
        self.plot_window = None  # Store window reference

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.init_splitter_tab()
        self.init_trainer_tab()
        self.init_predictor_tab()

    def init_splitter_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.split_input_path = QLineEdit()
        self.split_input_path.setPlaceholderText("Select CSV Dataset...")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(lambda: self.browse_file(self.split_input_path))
        file_layout.addWidget(self.split_input_path)
        file_layout.addWidget(btn_browse)

        form_layout = QFormLayout()
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.01, 0.99)
        self.spin_test_size.setValue(0.25)
        self.spin_test_size.setSingleStep(0.05)

        self.check_shuffle = QCheckBox("Shuffle Data");
        self.check_shuffle.setChecked(True)
        self.spin_seed = QSpinBox();
        self.spin_seed.setValue(42)

        form_layout.addRow("Test Size:", self.spin_test_size)
        form_layout.addRow("", self.check_shuffle)
        form_layout.addRow("Seed:", self.spin_seed)

        btn_process = QPushButton("Split and Save")
        btn_process.clicked.connect(self.run_splitter)

        layout.addWidget(QLabel("<h2>1. Dataset Splitter</h2>"))
        layout.addLayout(file_layout)
        layout.addLayout(form_layout)
        layout.addWidget(btn_process)
        layout.addStretch()

        # FIX: Set Layout to Tab
        tab.setLayout(layout)

        self.tabs.addTab(tab, "Data Splitter")

    def run_splitter(self):
        path = self.split_input_path.text()
        if not os.path.exists(path): return
        try:
            df = pd.read_csv(path, header=None, index_col=None)
            test, train = train_test_split(df, test_size=self.spin_test_size.value(),
                                           random_state=self.spin_seed.value(), shuffle=self.check_shuffle.isChecked())
            d = os.path.dirname(path)
            t_path = os.path.join(d, "train_set_gui.csv")
            v_path = os.path.join(d, "test_set_gui.csv")
            train.to_csv(t_path, header=False, index=False)
            test.to_csv(v_path, header=False, index=False)
            QMessageBox.information(self, "Success", "Files saved.")
            self.train_input_path.setText(t_path)
            self.predict_data_path.setText(v_path)
            self.detected_input_size = train.shape[1] - 2
            self.update_visualizer()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def init_trainer_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # File
        file_layout = QHBoxLayout()
        self.train_input_path = QLineEdit()
        self.train_input_path.setPlaceholderText("Select Training CSV...")
        self.train_input_path.textChanged.connect(self.load_input_dims)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(lambda: self.browse_file(self.train_input_path))
        file_layout.addWidget(self.train_input_path)
        file_layout.addWidget(btn_browse)

        # Params
        params_group = QGroupBox("Global Parameters")
        params_form = QFormLayout()
        self.spin_epochs = QSpinBox();
        self.spin_epochs.setRange(1, 1000000);
        self.spin_epochs.setValue(1000)
        self.spin_lr = QDoubleSpinBox();
        self.spin_lr.setRange(0.0001, 1.0);
        self.spin_lr.setDecimals(4);
        self.spin_lr.setValue(0.001)
        self.spin_batch = QSpinBox();
        self.spin_batch.setRange(1, 512);
        self.spin_batch.setValue(32)
        self.spin_patience = QSpinBox();
        self.spin_patience.setRange(1, 1000000);
        self.spin_patience.setValue(50)
        params_form.addRow("Epochs:", self.spin_epochs)
        params_form.addRow("Learn Rate:", self.spin_lr)
        params_form.addRow("Batch Size:", self.spin_batch)
        params_form.addRow("Patience:", self.spin_patience)
        params_group.setLayout(params_form)

        # Layers
        layer_group = QGroupBox("Hidden Layers")
        layer_layout = QVBoxLayout()
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_container = QWidget()
        self.l_layout = QVBoxLayout(self.layer_container)
        self.l_layout.addStretch()
        self.layer_scroll.setWidget(self.layer_container)
        self.layer_rows = []
        btn_add = QPushButton("+ Add Layer")
        btn_add.clicked.connect(self.add_layer_row)
        layer_layout.addWidget(self.layer_scroll)
        layer_layout.addWidget(btn_add)
        layer_group.setLayout(layer_layout)

        # Output
        out_group = QGroupBox("Output Layer")
        out_layout = QHBoxLayout()
        self.spin_out_size = QSpinBox();
        self.spin_out_size.setValue(2)
        self.spin_out_size.valueChanged.connect(self.update_visualizer)
        self.combo_out_act = QComboBox();
        self.combo_out_act.addItems(["softmax", "sigmoid", "relu"])
        out_layout.addWidget(QLabel("Neurons:"))
        out_layout.addWidget(self.spin_out_size)
        out_layout.addWidget(QLabel("Act:"))
        out_layout.addWidget(self.combo_out_act)
        out_group.setLayout(out_layout)

        # Actions
        self.chk_bin = QCheckBox("Label Binarizer");
        self.chk_bin.setChecked(True)
        self.chk_sca = QCheckBox("Standard Scaler");
        self.chk_sca.setChecked(True)
        self.chk_var = QCheckBox("Variance Threshold")

        self.btn_train = QPushButton("Start Training")
        self.btn_train.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.btn_train.clicked.connect(self.start_training)

        # New Plot Button
        self.btn_show_plots = QPushButton("Show Training Plots")
        self.btn_show_plots.setEnabled(False)
        self.btn_show_plots.clicked.connect(self.open_plot_window)

        self.btn_save = QPushButton("Save Model")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_model)

        left_layout.addWidget(QLabel("<h2>2. Model Builder</h2>"))
        left_layout.addLayout(file_layout)
        left_layout.addWidget(params_group)
        left_layout.addWidget(layer_group, 1)
        left_layout.addWidget(out_group)
        left_layout.addWidget(self.chk_bin)
        left_layout.addWidget(self.chk_sca)
        left_layout.addWidget(self.chk_var)
        left_layout.addWidget(self.btn_train)
        left_layout.addWidget(self.btn_show_plots)
        left_layout.addWidget(self.btn_save)

        # Visualizer
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.net_viz = NetworkVisualizer()
        self.lbl_status = QLabel("Status: Ready")
        right_layout.addWidget(QLabel("<b>Network Architecture</b>"))
        right_layout.addWidget(self.net_viz)
        right_layout.addWidget(self.lbl_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 750])
        main_layout.addWidget(splitter)

        # FIX: Set Layout to Tab
        tab.setLayout(main_layout)

        self.tabs.addTab(tab, "Train Model")

        self.detected_input_size = 2
        self.add_layer_row(30, "Hidden 1")
        self.add_layer_row(30, "Hidden 2")
        self.trained_pipeline = None

    def add_layer_row(self, n=30, name=None):
        row = LayerConfigRow(len(self.layer_rows) + 1)
        if name: row.name_edit.setText(name)
        row.neurons_spin.setValue(n)
        row.removed.connect(self.remove_layer_row)
        row.changed.connect(self.update_visualizer)
        self.l_layout.insertWidget(len(self.layer_rows), row)
        self.layer_rows.append(row)
        self.update_visualizer()

    def remove_layer_row(self, w):
        w.deleteLater()
        self.layer_rows.remove(w)
        for i, r in enumerate(self.layer_rows): r.index_lbl.setText(f"L{i + 1}:")
        self.update_visualizer()

    def load_input_dims(self):
        path = self.train_input_path.text()
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, header=None, nrows=2)
                cols = df.shape[1]
                self.detected_input_size = cols - 2 if cols > 2 else cols
                y = pd.read_csv(path, header=None, usecols=[1]).values.flatten()
                u = len(np.unique(y))
                if u > 2:
                    self.spin_out_size.setValue(u)
                    self.combo_out_act.setCurrentText("softmax")
                self.update_visualizer()
            except:
                pass

    def update_visualizer(self):
        cfgs = [r.get_config() for r in self.layer_rows]
        self.net_viz.set_structure(self.detected_input_size, cfgs, self.spin_out_size.value())

    def start_training(self):
        path = self.train_input_path.text()
        if not os.path.exists(path): return
        try:
            df = pd.read_csv(path, header=None)
            X = df.values[:, 2:].astype(float)
            y = df.values[:, 1]

            l_cfg = []
            for r in self.layer_rows:
                c = r.get_config()
                l_cfg.append(Layers(c['count'], activation=c['activation'], name=c['name']))
            l_cfg.append(Layers(self.spin_out_size.value(), activation=self.combo_out_act.currentText(), name="Output"))

            mlp = MLP(tuple(l_cfg), epochs=self.spin_epochs.value(),
                      learning_rate=self.spin_lr.value(), batch_size=self.spin_batch.value(),
                      patience=self.spin_patience.value())

            # Save reference for plotting
            self.current_mlp = mlp

            steps = []
            if self.chk_bin.isChecked(): steps.append(LabelBinarizer())
            if self.chk_sca.isChecked(): steps.append(StandardScaler())
            if self.chk_var.isChecked(): steps.append(VarianceThreshold())
            steps.append(mlp)

            pipe = make_pipeline(*steps)

            self.lbl_status.setText("Training...")
            self.btn_train.setEnabled(False)
            self.btn_show_plots.setEnabled(True)  # Enable plot button
            self.net_viz.start_animation()

            self.worker = TrainingWorker(pipe, X, y)
            self.worker.finished.connect(self.training_done)
            self.worker.error.connect(lambda e: self.training_err(e))
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def training_done(self, pipe):
        self.net_viz.stop_animation()
        self.trained_pipeline = pipe
        self.lbl_status.setText("Complete!")
        self.btn_train.setEnabled(True)
        self.btn_save.setEnabled(True)
        QMessageBox.information(self, "Done", "Training Finished.")

    def training_err(self, e):
        self.net_viz.stop_animation()
        self.btn_train.setEnabled(True)
        self.lbl_status.setText("Error")
        QMessageBox.critical(self, "Error", e)

    def open_plot_window(self):
        if not self.current_mlp: return
        self.plot_window = PlotWindow(self.current_mlp)
        self.plot_window.show()

    def save_model(self):
        if not self.trained_pipeline: return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "", "Pickle (*.pkl)")
        if p:
            save_in_file(p, self.trained_pipeline)
            self.predict_model_path.setText(p)

    def init_predictor_tab(self):
        tab = QWidget()
        layout = QHBoxLayout()
        left = QWidget();
        l_lay = QVBoxLayout(left)

        m_lay = QHBoxLayout()
        self.predict_model_path = QLineEdit();
        btn_m = QPushButton("Browse")
        btn_m.clicked.connect(lambda: self.browse_file(self.predict_model_path, "*.pkl"))
        m_lay.addWidget(self.predict_model_path);
        m_lay.addWidget(btn_m)

        d_lay = QHBoxLayout()
        self.predict_data_path = QLineEdit();
        btn_d = QPushButton("Browse")
        btn_d.clicked.connect(lambda: self.browse_file(self.predict_data_path))
        d_lay.addWidget(self.predict_data_path);
        d_lay.addWidget(btn_d)

        btn_cols = QPushButton("Load Columns")
        btn_cols.clicked.connect(self.load_pred_cols)

        self.col_table = QTableWidget()
        self.col_table.setColumnCount(3)
        self.col_table.setHorizontalHeaderLabels(["Index", "Sample", "Role"])
        self.col_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        btn_run = QPushButton("Run Prediction")
        btn_run.clicked.connect(self.run_pred)

        l_lay.addWidget(QLabel("<h2>3. Prediction</h2>"))
        l_lay.addWidget(QLabel("Model:"));
        l_lay.addLayout(m_lay)
        l_lay.addWidget(QLabel("Data:"));
        l_lay.addLayout(d_lay)
        l_lay.addWidget(btn_cols);
        l_lay.addWidget(self.col_table);
        l_lay.addWidget(btn_run)

        right = QWidget();
        r_lay = QVBoxLayout(right)
        self.lbl_acc = QLabel("Accuracy: N/A")
        self.res_table = QTableWidget()
        self.res_table.setColumnCount(3)
        self.res_table.setHorizontalHeaderLabels(["ID", "Actual", "Predicted"])
        self.res_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        r_lay.addWidget(QLabel("<b>Results</b>"));
        r_lay.addWidget(self.lbl_acc);
        r_lay.addWidget(self.res_table)

        split = QSplitter(Qt.Orientation.Horizontal);
        split.addWidget(left);
        split.addWidget(right)
        split.setSizes([400, 600])
        layout.addWidget(split)

        # FIX: Set Layout to Tab
        tab.setLayout(layout)

        self.tabs.addTab(tab, "Predict")

    def load_pred_cols(self):
        p = self.predict_data_path.text()
        if not os.path.exists(p): return
        try:
            df = pd.read_csv(p, header=None, nrows=5)
            self.col_table.setRowCount(df.shape[1])
            for i in range(df.shape[1]):
                self.col_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.col_table.setItem(i, 1, QTableWidgetItem(str(df.iloc[0, i])))
                cb = QComboBox();
                cb.addItems(["Feature (X)", "Target (y)", "Ignore"])
                if i == 0:
                    cb.setCurrentText("Ignore")
                elif i == 1:
                    cb.setCurrentText("Target (y)")
                else:
                    cb.setCurrentText("Feature (X)")
                self.col_table.setCellWidget(i, 2, cb)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_pred(self):
        if self.col_table.rowCount() == 0: return
        xi, yi = [], None
        for i in range(self.col_table.rowCount()):
            role = self.col_table.cellWidget(i, 2).currentText()
            if role == "Feature (X)":
                xi.append(i)
            elif role == "Target (y)":
                yi = i
        if not xi: return
        try:
            pipe = load_from_file(self.predict_model_path.text())
            df = pd.read_csv(self.predict_data_path.text(), header=None)
            X = df.iloc[:, xi].values.astype(float)
            y = df.iloc[:, yi].values if yi is not None else None
            pred = pipe.predict(X)
            self.res_table.setRowCount(len(pred))
            corr = 0
            for i, p in enumerate(pred):
                self.res_table.setItem(i, 0, QTableWidgetItem(str(i)))
                pv = np.argmax(p) if isinstance(p, np.ndarray) else p
                self.res_table.setItem(i, 2, QTableWidgetItem(str(pv)))
                if y is not None:
                    av = y[i]
                    self.res_table.setItem(i, 1, QTableWidgetItem(str(av)))
                    if av == pv:
                        corr += 1
                        self.res_table.item(i, 2).setBackground(Qt.GlobalColor.green)
                    else:
                        self.res_table.item(i, 2).setBackground(Qt.GlobalColor.red)
                else:
                    self.res_table.setItem(i, 1, QTableWidgetItem("-"))
            if y is not None: self.lbl_acc.setText(f"Accuracy: {(corr / len(pred)) * 100:.2f}%")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def browse_file(self, line, f="CSV (*.csv)"):
        p, _ = QFileDialog.getOpenFileName(self, "Select", "", f)
        if p: line.setText(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MLPGUI()
    w.show()
    sys.exit(app.exec())