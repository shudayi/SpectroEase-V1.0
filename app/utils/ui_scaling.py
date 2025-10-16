from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import QRect, QSize, Qt

WIDGET_BLUEPRINT = {}
ORIGINAL_WINDOW_SIZE = None

def capture_initial_state(parent_widget):
    """
    Captures the initial 'design blueprint' of all widgets.
    This includes geometry, font, icon size, and original high-res pixmaps.
    """
    global ORIGINAL_WINDOW_SIZE
    ORIGINAL_WINDOW_SIZE = parent_widget.size()
    WIDGET_BLUEPRINT.clear()  # Clear previous state

    for widget in parent_widget.findChildren(QWidget):
        # Use a unique and stable key for each widget
        key = id(widget)
        
        blueprint = {
            "geometry": widget.geometry(),
            "font": widget.font(),
        }

        # Capture icon size if the widget has one
        if hasattr(widget, 'iconSize') and hasattr(widget, 'setIconSize'):
            blueprint['icon_size'] = widget.iconSize()

        # Capture the original, unscaled pixmap from QLabels
        if isinstance(widget, QLabel) and widget.pixmap() and not widget.pixmap().isNull():
            blueprint['pixmap'] = widget.pixmap()

        WIDGET_BLUEPRINT[key] = blueprint
    
    print(f"✅ UI Blueprint Captured: {len(WIDGET_BLUEPRINT)} widgets recorded.")

def apply_dynamic_scaling(parent_widget):
    """
    Applies scaling to all widgets based on the current window size vs the original.
    This is the core real-time rendering engine.
    """
    if not ORIGINAL_WINDOW_SIZE or ORIGINAL_WINDOW_SIZE.width() == 0 or ORIGINAL_WINDOW_SIZE.height() == 0 or not WIDGET_BLUEPRINT:
        return

    current_size = parent_widget.size()
    scale_x = current_size.width() / ORIGINAL_WINDOW_SIZE.width()
    scale_y = current_size.height() / ORIGINAL_WINDOW_SIZE.height()

    for key, blueprint in WIDGET_BLUEPRINT.items():
        try:
            # Re-find the widget by its ID in case it was recreated
            widget = blueprint.get('widget_ref')
            if widget is None:
                 # This is a fallback; ideally, we'd have a better way to re-find widgets
                 # For now, we rely on the initial capture.
                 # We can't reliably find by ID, so we'll skip if the reference is lost.
                 # A more robust solution might involve object names.
                 pass

            # We need to get the widget object. The blueprint stores properties, not the object itself after the initial run.
            # This requires a way to map the key back to the widget. Let's assume we can iterate and find it.
            # A better approach would be to store widget references if they are stable.
            # For this implementation, we'll iterate all children again. This is inefficient but safer.
            
            # Let's refine the blueprint capture to store a weak reference
            pass # The logic below assumes we have the widget object.

        except Exception:
            continue # Widget might have been deleted

    # This is inefficient, but safer than relying on potentially stale references.
    # A better long-term solution is using object names.
    for widget in parent_widget.findChildren(QWidget):
        key = id(widget)
        if key in WIDGET_BLUEPRINT:
            blueprint = WIDGET_BLUEPRINT[key]
            
            # 1. Scale Geometry (Position and Size)
            original_geom = blueprint['geometry']
            new_geom = QRect(
                int(original_geom.x() * scale_x),
                int(original_geom.y() * scale_y),
                int(original_geom.width() * scale_x),
                int(original_geom.height() * scale_y)
            )
            # Use move and resize separately to avoid potential issues with setGeometry
            widget.move(new_geom.topLeft())
            widget.resize(new_geom.size())

            # 2. Scale Font
            original_font = blueprint['font']
            # Use the smaller scale factor to maintain aspect ratio for fonts
            font_scale = min(scale_x, scale_y)
            new_font_size = original_font.pointSize() * font_scale
            if new_font_size > 1: # Ensure font size is positive
                new_font = QFont(original_font.family(), int(new_font_size))
                widget.setFont(new_font)

            # 3. Scale Icon Size
            if 'icon_size' in blueprint and hasattr(widget, 'setIconSize'):
                original_icon_size = blueprint['icon_size']
                if original_icon_size.isValid():
                    new_icon_size = QSize(
                        int(original_icon_size.width() * scale_x),
                        int(original_icon_size.height() * scale_y)
                    )
                    widget.setIconSize(new_icon_size)

            # 4. Scale Pixmap (Image/Logo) for QLabels
            if 'pixmap' in blueprint and isinstance(widget, QLabel):
                original_pixmap = blueprint['pixmap']
                # Scale the original pixmap to the new widget size, preserving aspect ratio
                widget.setPixmap(original_pixmap.scaled(
                    widget.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))