import sys


def log(*msgs, lvl=0, quiet=False):
    """Format and print message to console with one of the following logging levels:
    0: info (print and continue execution; ignore if quiet=True)
    1: warning (print and continue execution)
    2: error (print and exit with code 1)
    """

    if lvl == 0 and quiet:
        return

    prefix = ""
    if lvl == 0:
        prefix = "INFO:"
    elif lvl == 1:
        prefix = "WARN:"
    elif lvl == 2:
        prefix = "CRITICAL:"

    print(f"{prefix}", *msgs)
    if lvl == 2:
        sys.exit(1)


def style_ax(ax, xlab="", ylab="", aspect="auto", xlim=None, ylim=None):
    for axis in ("top", "bottom", "left", "right"):
        # hide axes
        if axis in ("top", "right"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

        # general formatting
        ax.spines[axis].set_linewidth(1.0)
        ax.set_xlabel(xlab, fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            direction="out",
            length=4.0,
            width=1.0,
            color="k",
            labelsize=12,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            direction="out",
            length=4.0,
            width=1.0,
            color="k",
            labelsize=12,
        )
        ax.grid(color="gray", ls=":", lw=0.5, alpha=0.5)

        # set aspect ratio
        # values can be "auto", "equal" (same as 1), or a float
        ax.set_aspect(aspect)

        # set x and y limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # useful for chaining (is ax modified in place?)
        return ax
