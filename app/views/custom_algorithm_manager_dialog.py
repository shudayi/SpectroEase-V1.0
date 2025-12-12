"""
Custom Algorithm Manager Dialog
UI for managing saved custom algorithms
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QMessageBox, QHeaderView, QTextEdit, QAbstractItemView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from typing import Dict, Callable
from app.views.responsive_dialog import ResponsiveDialog


class CustomAlgorithmManagerDialog(ResponsiveDialog):
    """Dialog for managing custom algorithms"""
    
    def __init__(self, cache_service, reload_callback: Callable = None, parent=None, remove_callback: Callable = None):
        """
        Initialize the manager dialog
        
        Args:
            cache_service: CustomAlgorithmCache instance
            reload_callback: Function to call when algorithms need to be reloaded
            parent: Parent widget
            remove_callback: Function to call when algorithm is deleted (algorithm_type, name)
        """
        super().__init__(parent, base_width=900, base_height=700)
        self.cache_service = cache_service
        self.remove_callback = remove_callback  # V1.4.1: Callback to remove algorithm from UI
        self.reload_callback = reload_callback
        self.init_ui()
        self.load_algorithms()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Manage Custom Algorithms")
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Custom Algorithm Manager")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Info label
        info = QLabel("View and manage your saved custom algorithms")
        info.setStyleSheet("color: #666; font-size: 10px;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        
        # Tab widget for different algorithm types
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Create tabs for each algorithm type
        self.algorithm_types = {
            'Preprocessing': 'preprocessing',
            'Feature Selection': 'feature_selection',
            'Modeling': 'modeling',
            'Data Partitioning': 'data_partitioning'
        }
        
        self.tables = {}
        for display_name, cache_key in self.algorithm_types.items():
            tab_widget = self._create_algorithm_tab(display_name, cache_key)
            self.tabs.addTab(tab_widget, display_name)
        
        layout.addWidget(self.tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.refresh_button.clicked.connect(self.load_algorithms)
        button_layout.addWidget(self.refresh_button)
        
        self.reload_all_button = QPushButton("Reload All to System")
        self.reload_all_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.reload_all_button.clicked.connect(self.reload_all_algorithms)
        button_layout.addWidget(self.reload_all_button)
        
        button_layout.addStretch()
        
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #757575; }
        """)
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _create_algorithm_tab(self, display_name: str, cache_key: str) -> QWidget:
        """Create a tab for a specific algorithm type"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Table to display algorithms
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Algorithm Name", "Created", "Updated", "Actions"])
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """)
        
        self.tables[cache_key] = table
        layout.addWidget(table)
        
        # Info label
        self.info_labels = getattr(self, 'info_labels', {})
        info_label = QLabel(f"No {display_name.lower()} algorithms saved")
        info_label.setStyleSheet("color: #999; font-style: italic; padding: 20px;")
        info_label.setAlignment(Qt.AlignCenter)
        self.info_labels[cache_key] = info_label
        layout.addWidget(info_label)
        
        return tab
    
    def load_algorithms(self):
        """Load and display all algorithms"""
        for display_name, cache_key in self.algorithm_types.items():
            table = self.tables[cache_key]
            info_label = self.info_labels[cache_key]
            
            # Clear table
            table.setRowCount(0)
            
            # Load algorithms from cache
            algorithms = self.cache_service.load_algorithms(cache_key)
            
            if algorithms:
                info_label.hide()
                table.show()
                
                # Populate table
                for algo in algorithms:
                    row = table.rowCount()
                    table.insertRow(row)
                    
                    # Name
                    name_item = QTableWidgetItem(algo.get('name', 'Unknown'))
                    table.setItem(row, 0, name_item)
                    
                    # Created date
                    created = algo.get('created_at', 'Unknown')
                    if isinstance(created, str) and 'T' in created:
                        created = created.split('T')[0]  # Show just the date
                    created_item = QTableWidgetItem(str(created))
                    table.setItem(row, 1, created_item)
                    
                    # Updated date
                    updated = algo.get('updated_at', 'Unknown')
                    if isinstance(updated, str) and 'T' in updated:
                        updated = updated.split('T')[0]  # Show just the date
                    updated_item = QTableWidgetItem(str(updated))
                    table.setItem(row, 2, updated_item)
                    
                    # Action buttons
                    action_widget = QWidget()
                    action_layout = QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(4, 4, 4, 4)
                    action_layout.setSpacing(4)
                    
                    view_btn = QPushButton("View Code")
                    view_btn.setFixedWidth(80)
                    view_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #2196F3;
                            color: white;
                            border: none;
                            padding: 4px 8px;
                            border-radius: 3px;
                            font-size: 10px;
                        }
                        QPushButton:hover { background-color: #1976D2; }
                    """)
                    view_btn.clicked.connect(
                        lambda checked, n=algo.get('name'), c=algo.get('code'): 
                        self.view_algorithm_code(n, c)
                    )
                    action_layout.addWidget(view_btn)
                    
                    delete_btn = QPushButton("Delete")
                    delete_btn.setFixedWidth(60)
                    delete_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #f44336;
                            color: white;
                            border: none;
                            padding: 4px 8px;
                            border-radius: 3px;
                            font-size: 10px;
                        }
                        QPushButton:hover { background-color: #da190b; }
                    """)
                    delete_btn.clicked.connect(
                        lambda checked, t=cache_key, n=algo.get('name'): 
                        self.delete_algorithm(t, n)
                    )
                    action_layout.addWidget(delete_btn)
                    
                    table.setCellWidget(row, 3, action_widget)
            else:
                table.hide()
                info_label.show()
    
    def view_algorithm_code(self, name: str, code: str):
        """Display algorithm code in a dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{name} - Code")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Code display
        code_edit = QTextEdit()
        code_edit.setPlainText(code)
        code_edit.setReadOnly(True)
        code_edit.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        layout.addWidget(code_edit)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #757575; }
        """)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def delete_algorithm(self, algorithm_type: str, name: str):
        """Delete an algorithm"""
        reply = QMessageBox.question(
            self,
            'Confirm Delete',
            f"Are you sure you want to delete the custom algorithm '{name}'?\n\n"
            "This will remove it from the cache. You will need to restart the application "
            "for changes to take full effect.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.cache_service.delete_algorithm(algorithm_type, name)
            
            if success:
                # V1.4.1: Immediately remove from UI if callback is provided
                if self.remove_callback:
                    try:
                        self.remove_callback(algorithm_type, name)
                        print(f"✅ Removed algorithm '{name}' from UI immediately")
                    except Exception as e:
                        print(f"⚠️  Failed to remove algorithm from UI: {e}")
                        import traceback
                        traceback.print_exc()
                
                QMessageBox.information(
                    self,
                    'Success',
                    f"Algorithm '{name}' has been deleted.\n\n"
                    + ("Algorithm has been removed from the UI." if self.remove_callback 
                       else "Please restart the application to fully remove it from the UI.")
                )
                self.load_algorithms()  # Refresh the display
            else:
                QMessageBox.warning(
                    self,
                    'Error',
                    f"Failed to delete algorithm '{name}'"
                )
    
    def reload_all_algorithms(self):
        """Reload all algorithms to the system"""
        if self.reload_callback:
            reply = QMessageBox.question(
                self,
                'Reload Algorithms',
                "This will reload all custom algorithms into the system.\n"
                "Any unsaved changes will be lost. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    self.reload_callback()
                    QMessageBox.information(
                        self,
                        'Success',
                        "All custom algorithms have been reloaded successfully."
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        'Error',
                        f"Failed to reload algorithms: {str(e)}"
                    )
        else:
            QMessageBox.warning(
                self,
                'Not Available',
                "Reload functionality is not available. Please restart the application."
            )





