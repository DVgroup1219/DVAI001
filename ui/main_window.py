"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from core.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


class AnalysisWorker(QObject):
    """Background worker for JPEG analysis."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, paths: list[Path], output_dir: Path) -> None:
        super().__init__()
        self._paths = paths
        self._output_dir = output_dir
        self._pipeline = AnalysisPipeline()

    @Slot()
    def run(self) -> None:
        """Execute analysis in a worker thread."""
        try:
            report = self._pipeline.analyze_paths(
                self._paths,
                self._output_dir,
                progress_callback=lambda cur, total, name: self.progress.emit(
                    cur, total, name
                ),
            )
            self.finished.emit(report)
        except Exception as exc:
            logger.exception("Analysis failed")
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    """Main UI for DV Studio Color AI - Module 01."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DV Studio Color AI — Module 01 — Sony JPEG Analyzer")
        self.resize(1100, 720)

        self._selected_paths: list[Path] = []
        self._output_dir = Path.cwd() / "cache" / "output"
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the main window layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Sony JPEG Analyzer — Read-Only Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select JPEG images or a folder. Analysis exports analysis.json and analysis.csv. "
            "Images are never modified."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        btn_row = QHBoxLayout()
        self._btn_single = QPushButton("Single Image")
        self._btn_multiple = QPushButton("Multiple Images")
        self._btn_folder = QPushButton("Folder")
        self._btn_output = QPushButton("Output Folder")
        self._btn_analyze = QPushButton("Start Analysis")
        self._btn_analyze.setStyleSheet("font-weight: bold;")

        self._btn_single.clicked.connect(self._select_single)
        self._btn_multiple.clicked.connect(self._select_multiple)
        self._btn_folder.clicked.connect(self._select_folder)
        self._btn_output.clicked.connect(self._select_output)
        self._btn_analyze.clicked.connect(self._start_analysis)

        for btn in (
            self._btn_single,
            self._btn_multiple,
            self._btn_folder,
            self._btn_output,
            self._btn_analyze,
        ):
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self._path_label = QLabel("No images selected")
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

        self._output_label = QLabel(f"Output: {self._output_dir}")
        self._output_label.setWordWrap(True)
        layout.addWidget(self._output_label)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._status = QLabel("Ready")
        layout.addWidget(self._status)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            [
                "File",
                "Outdoor",
                "Indoor",
                "Faces",
                "Skin Tone",
                "Highlight %",
                "Shadow %",
                "Dominant Color",
            ]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        layout.addWidget(self._log)

    def _select_single(self) -> None:
        """Select a single JPEG file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JPEG Image",
            "",
            "JPEG Images (*.jpg *.jpeg *.JPG *.JPEG)",
        )
        if path:
            self._selected_paths = [Path(path)]
            self._update_path_label()

    def _select_multiple(self) -> None:
        """Select multiple JPEG files."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select JPEG Images",
            "",
            "JPEG Images (*.jpg *.jpeg *.JPG *.JPEG)",
        )
        if paths:
            self._selected_paths = [Path(p) for p in paths]
            self._update_path_label()

    def _select_folder(self) -> None:
        """Select a folder containing JPEG files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            from utils.file_scanner import FileScanner

            scanner = FileScanner()
            self._selected_paths = scanner.scan_folder(Path(folder))
            self._update_path_label()

    def _select_output(self) -> None:
        """Select output directory for analysis files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_dir = Path(folder)
            self._output_label.setText(f"Output: {self._output_dir}")

    def _update_path_label(self) -> None:
        """Refresh selected paths label."""
        count = len(self._selected_paths)
        if count == 1:
            self._path_label.setText(f"Selected: {self._selected_paths[0]}")
        elif count > 1:
            self._path_label.setText(f"Selected: {count} JPEG files")
        else:
            self._path_label.setText("No images selected")

    def _start_analysis(self) -> None:
        """Start background analysis."""
        if not self._selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select JPEG images first.")
            return

        self._set_ui_busy(True)
        self._progress.setMaximum(len(self._selected_paths))
        self._progress.setValue(0)
        self._log.clear()
        self._append_log(f"Starting analysis of {len(self._selected_paths)} image(s)...")

        self._thread = QThread()
        self._worker = AnalysisWorker(self._selected_paths, self._output_dir)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_progress(self, current: int, total: int, name: str) -> None:
        """Update progress bar and status."""
        self._progress.setValue(current)
        self._status.setText(f"Analyzing {current}/{total}: {name}")
        self._append_log(f"Analyzed: {name}")

    def _on_finished(self, report) -> None:
        """Handle completed analysis."""
        self._set_ui_busy(False)
        self._populate_table(report)
        self._status.setText(
            f"Complete — {len(report.images)} image(s) analyzed. "
            f"Saved to {self._output_dir}"
        )
        self._append_log(f"Exported analysis.json and analysis.csv to {self._output_dir}")
        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analysis complete.\n\nOutput:\n{self._output_dir / 'analysis.json'}\n"
            f"{self._output_dir / 'analysis.csv'}",
        )

    def _on_error(self, message: str) -> None:
        """Handle analysis errors."""
        self._set_ui_busy(False)
        self._status.setText("Error")
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Analysis Error", message)

    def _populate_table(self, report) -> None:
        """Fill results table from analysis report."""
        self._table.setRowCount(len(report.images))
        for row, img in enumerate(report.images):
            row_data = [
                img.file_name,
                str(img.scene.outdoor),
                str(img.scene.indoor),
                str(len(img.faces)),
                img.color.skin_tone_label,
                f"{img.statistics.highlight_percentage:.1f}",
                f"{img.statistics.shadow_percentage:.1f}",
                img.background.dominant_color,
            ]
            for col, value in enumerate(row_data):
                self._table.setItem(row, col, QTableWidgetItem(value))

    def _set_ui_busy(self, busy: bool) -> None:
        """Enable or disable controls during analysis."""
        for btn in (
            self._btn_single,
            self._btn_multiple,
            self._btn_folder,
            self._btn_output,
            self._btn_analyze,
        ):
            btn.setEnabled(not busy)

    def _append_log(self, message: str) -> None:
        """Append a line to the log panel."""
        self._log.append(message)


def run_app() -> None:
    """Launch the PySide6 application."""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
