"""
utils.py — shared matplotlib helpers for viz modules
"""


def multicolor_text(fig, ax, segments, y, fontsize=13, fontweight='bold',
                    ref='axes_left'):
    """
    Render consecutive coloured text segments on a single line.

    Parameters
    ----------
    fig, ax   : matplotlib Figure and Axes
    segments  : list of (text, colour) tuples
    y         : y position in figure fraction (0–1)
    fontsize, fontweight : text style
    ref       : 'axes_left' — left-align to the left edge of ax
                'center'    — centre the whole line over ax
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w    = fig.get_window_extent(renderer).width
    ax_bbox  = ax.get_window_extent(renderer)

    # Phase 1: render all segments at x=0 to measure widths
    rendered = []
    x_px     = 0.0
    for text, colour in segments:
        t = fig.text(0.0, y, text, ha='left', va='bottom',
                     fontsize=fontsize, fontweight=fontweight, color=colour)
        fig.canvas.draw()
        w = t.get_window_extent(renderer).width
        rendered.append((t, w))
        x_px += w

    total_w = x_px

    # Phase 2: compute start x in pixels then reposition
    if ref == 'center':
        ax_cx   = (ax_bbox.x0 + ax_bbox.x1) / 2
        start_x = ax_cx - total_w / 2
    else:
        start_x = ax_bbox.x0

    x_px = start_x
    for t, w in rendered:
        t.set_position((x_px / fig_w, y))
        x_px += w
