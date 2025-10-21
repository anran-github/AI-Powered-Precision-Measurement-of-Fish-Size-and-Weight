# fish_labeler.py
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt
import cv2
import os
import sys
import numpy as np
from glob import glob

# ---- User-editable paths ----
IMAGE_FOLDER = "images"   # folder with RGB images (png/jpg)
OUT_HEAD = "masks/head"   # saved masks: one file per class, same filename as image
OUT_BODY = "masks/body"
OUT_TAIL = "masks/tail"
OUT_FINS = "masks/fins"
os.makedirs(OUT_HEAD, exist_ok=True)
os.makedirs(OUT_BODY, exist_ok=True)
os.makedirs(OUT_TAIL, exist_ok=True)
os.makedirs(OUT_FINS, exist_ok=True)

CLASS_MAP = {
    1: ("head", OUT_HEAD, (255, 0, 0, 60)),    # red-ish
    2: ("body", OUT_BODY, (0, 255, 0, 60)),    # green-ish
    3: ("tail", OUT_TAIL, (0, 0, 255, 60)),    # blue-ish
    4: ("fins", OUT_FINS, (255, 255, 0, 60)),  # yellow-ish
}

# ------------------------------

class LabelerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fish Section Labeler")
        self.resize(1100, 700)

        # central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        # Left: image display
        self.imageLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.imageLabel.setMinimumSize(640, 480)
        self.imageLabel.setStyleSheet("background-color: #222;")
        self.imageLabel.setScaledContents(True)
        hbox.addWidget(self.imageLabel, 1)

        # Right: controls
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        hbox.addWidget(controls, 0)

        # Info
        self.statusText = QtWidgets.QLineEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setPlaceholderText("Status...")
        controls_layout.addWidget(self.statusText)

        # Class buttons
        self.classButtons = {}
        for cls in sorted(CLASS_MAP.keys()):
            name = CLASS_MAP[cls][0]
            btn = QtWidgets.QPushButton(f"{cls}: {name}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=cls: self.select_class(s))
            controls_layout.addWidget(btn)
            self.classButtons[cls] = btn

        # brush controls
        brush_row = QtWidgets.QHBoxLayout()
        self.brushLabel = QtWidgets.QLabel("Brush:")
        brush_row.addWidget(self.brushLabel)
        self.brushSizeLabel = QtWidgets.QLabel("10")
        brush_row.addWidget(self.brushSizeLabel)
        controls_layout.addLayout(brush_row)

        # navigation
        nav_row = QtWidgets.QHBoxLayout()
        self.prevBtn = QtWidgets.QPushButton("◀ Prev")
        self.prevBtn.clicked.connect(self.prev_image)
        nav_row.addWidget(self.prevBtn)
        self.nextBtn = QtWidgets.QPushButton("Next ▶")
        self.nextBtn.clicked.connect(self.next_image)
        nav_row.addWidget(self.nextBtn)
        controls_layout.addLayout(nav_row)

        jump_row = QtWidgets.QHBoxLayout()
        self.jumpLineEdit = QtWidgets.QLineEdit()
        self.jumpLineEdit.setPlaceholderText("Jump to #")
        self.jumpLineEdit.returnPressed.connect(self.jump_to_frame)
        jump_row.addWidget(self.jumpLineEdit)
        self.loadFolderBtn = QtWidgets.QPushButton("Load Folder")
        self.loadFolderBtn.clicked.connect(self.select_folder)
        jump_row.addWidget(self.loadFolderBtn)
        controls_layout.addLayout(jump_row)

        # save / other
        self.saveBtn = QtWidgets.QPushButton("Save (S)")
        self.saveBtn.clicked.connect(self.save_current_mask)
        controls_layout.addWidget(self.saveBtn)

        controls_layout.addStretch(1)

        # shortcuts
        self.shortcut_plus = QtWidgets.QShortcut(QtGui.QKeySequence("+"), self)
        self.shortcut_minus = QtWidgets.QShortcut(QtGui.QKeySequence("-"), self)
        self.shortcut_plus.activated.connect(lambda: self.change_brush(+2))
        self.shortcut_minus.activated.connect(lambda: self.change_brush(-2))

        self.shortcut_undo = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_last_action)

        # class shortcuts 1-4
        for i in range(1, 5):
            QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), self).activated.connect(lambda s=i: self.select_class(s))

        # save shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self).activated.connect(self.save_current_mask)

        # internals
        self.datasetList = []
        self.number_dataset = 0
        self.currentFrame = 0
        self.currentImage = None        # filepath
        self.orig_img = None           # HxWx3 RGB numpy
        self.current_mask = None       # HxW integer (0..4)
        self.circle_radius = 10
        self.label_to_draw = 1
        self.subtract_mode = False
        self.drawing = False
        self.last_pos = None
        self.mask_history = []
        self.MAX_HISTORY = 25

        # connect events
        self.imageLabel.installEventFilter(self)

        # initial UI
        self.update_brush_label()
        self.select_class(1)
        # load images from IMAGE_FOLDER by default if exists
        if os.path.isdir(IMAGE_FOLDER):
            self.load_images_from_folder(IMAGE_FOLDER)
        else:
            self.statusText.setPlaceholderText("No default image folder. Click Load Folder.")

    # ---------- folder / dataset ----------
    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select image folder", ".")
        if folder:
            self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder):
        exts = ("*.png", "*.jpg", "*.JPG","*.JPEG", "*.jpeg", "*.bmp")
        dataset = []
        for e in exts:
            dataset.extend(glob(os.path.join(folder, e)))
        dataset = sorted(dataset)
        if not dataset:
            self.statusText.setPlaceholderText("No images found in folder.")
            return
        self.datasetList = dataset
        self.number_dataset = len(dataset)
        self.currentFrame = 0
        self.statusText.setPlaceholderText(f"Loaded {self.number_dataset} images.")
        self.load_current_image()

    def load_current_image(self):
        if not self.datasetList:
            return
        self.currentImage = self.datasetList[self.currentFrame]
        img = cv2.imread(self.currentImage)
        img = cv2.resize(img,(img.shape[1]//3, img.shape[0]//3))  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.orig_img = img
        h, w = img.shape[:2]
        # try to load existing masks (one per class) and combine to single mask
        mask = np.zeros((h, w), dtype=np.uint8)
        base = os.path.basename(self.currentImage)
        for cls, (name, outdir, _) in CLASS_MAP.items():
            p = os.path.join(outdir, base)
            if os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == (h, w):
                    mask[m != 0] = cls
        self.current_mask = mask
        self.mask_history = []
        self.update_overlay_display()
        self.statusText.setPlaceholderText(f"{os.path.basename(self.currentImage)} ({self.currentFrame + 1}/{self.number_dataset})")

    def next_image(self):
        if not self.datasetList:
            return
        if self.currentFrame < self.number_dataset - 1:
            self.currentFrame += 1
            self.load_current_image()

    def prev_image(self):
        if not self.datasetList:
            return
        if self.currentFrame > 0:
            self.currentFrame -= 1
            self.load_current_image()

    def jump_to_frame(self):
        if not self.datasetList:
            return
        text = self.jumpLineEdit.text().strip()
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < self.number_dataset:
                self.currentFrame = idx
                self.load_current_image()
                self.jumpLineEdit.clear()
            else:
                self.statusText.setPlaceholderText("Index out of range")
        else:
            self.statusText.setPlaceholderText("Invalid number")

    # ---------- class / brush ----------
    def select_class(self, cls):
        # toggle buttons
        for k, b in self.classButtons.items():
            b.setChecked(k == cls)
        self.label_to_draw = cls
        name = CLASS_MAP[cls][0]
        self.statusText.setPlaceholderText(f"Selected class {cls}: {name}")
        self.update_overlay_display()

    def change_brush(self, delta):
        self.circle_radius = max(2, self.circle_radius + delta)
        self.update_brush_label()

    def update_brush_label(self):
        self.brushSizeLabel.setText(str(self.circle_radius))

    # ---------- painting ----------
    def eventFilter(self, obj, event):
        if obj is self.imageLabel:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if self.current_mask is None:
                    return False
                if event.button() == QtCore.Qt.LeftButton:
                    # save history before new stroke
                    if self.current_mask is not None:
                        self.mask_history.append(self.current_mask.copy())
                        if len(self.mask_history) > self.MAX_HISTORY:
                            self.mask_history.pop(0)
                    self.drawing = True
                    self.last_pos = event.pos()
                    self.subtract_mode = (event.modifiers() & QtCore.Qt.ShiftModifier) != 0
                    self.draw_at(event.pos())
                    return True
            elif event.type() == QtCore.QEvent.MouseMove and getattr(self, "drawing", False):
                self.draw_line(self.last_pos, event.pos())
                self.last_pos = event.pos()
                return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if getattr(self, "drawing", False):
                    self.drawing = False
                    return True
        return super().eventFilter(obj, event)

    def draw_at(self, pos):
        h, w = self.current_mask.shape
        x = int(pos.x() * w / self.imageLabel.width())
        y = int(pos.y() * h / self.imageLabel.height())
        self.modify_mask_at_point(x, y)
        self.update_overlay_display()

    def draw_line(self, start_pos, end_pos):
        if self.current_mask is None:
            return
        w = self.current_mask.shape[1]
        h = self.current_mask.shape[0]
        x1 = int(start_pos.x() * w / self.imageLabel.width())
        y1 = int(start_pos.y() * h / self.imageLabel.height())
        x2 = int(end_pos.x() * w / self.imageLabel.width())
        y2 = int(end_pos.y() * h / self.imageLabel.height())
        num = max(abs(x2 - x1), abs(y2 - y1)) + 1
        xs = np.linspace(x1, x2, num).astype(np.int32)
        ys = np.linspace(y1, y2, num).astype(np.int32)
        for x, y in zip(xs, ys):
            self.modify_mask_at_point(int(x), int(y))
        self.update_overlay_display()

    def modify_mask_at_point(self, x, y):
        if self.current_mask is None:
            return
        yy, xx = np.ogrid[:self.current_mask.shape[0], :self.current_mask.shape[1]]
        mask_area = (xx - x) ** 2 + (yy - y) ** 2 <= self.circle_radius ** 2
        if self.subtract_mode:
            # set to 0
            self.current_mask[mask_area] = 0
        else:
            self.current_mask[mask_area] = self.label_to_draw

    def undo_last_action(self):
        if self.mask_history:
            self.current_mask = self.mask_history.pop()
            self.update_overlay_display()
            self.statusText.setPlaceholderText("Undo performed")
        else:
            self.statusText.setPlaceholderText("Nothing to undo")

    # ---------- overlay & saving ----------
    def mask_to_overlay(self, orig_img, mask):
        """Return PIL RGBA image that is alpha-composited overlay of mask on orig_img."""
        # orig_img: HxWx3 uint8 RGB
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        # Compose overlay for ALL labels (so multi-colored mask is visible)
        for cls, (_, _, color) in CLASS_MAP.items():
            r, g, b, a = color
            overlay[mask == cls] = (r, g, b, a)
        overlay_img = Image.fromarray(overlay, "RGBA")
        orig_pil = Image.fromarray(orig_img).convert("RGBA")
        comp = Image.alpha_composite(orig_pil, overlay_img)
        return comp

    def update_overlay_display(self):
        if self.orig_img is None:
            return
        overlay = self.mask_to_overlay(self.orig_img, self.current_mask)
        qt_img = ImageQt.ImageQt(overlay)
        pix = QtGui.QPixmap.fromImage(qt_img)
        self.imageLabel.setPixmap(pix)
        self.imageLabel.setScaledContents(True)

    def save_current_mask(self):
        if self.current_mask is None or self.currentImage is None:
            self.statusText.setPlaceholderText("Nothing to save")
            return
        base = os.path.basename(self.currentImage)
        h, w = self.current_mask.shape
        # for each class, save binary mask (0/255) as PNG in its folder
        for cls, (name, outdir, _) in CLASS_MAP.items():
            out_path = os.path.join(outdir, base)
            mask_bin = np.zeros((h, w), dtype=np.uint8)
            mask_bin[self.current_mask == cls] = 255
            cv2.imwrite(out_path, mask_bin)
        self.statusText.setPlaceholderText(f"Saved masks for {base}")
        print(f"Saved masks for {base}")

# --------- run ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LabelerWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
