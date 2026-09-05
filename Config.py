runName = "baseline"

paramMode = "fixed"        # "fixed" freezes alpha/beta/mean/std at formation, "rolling" is the old 60-day window

useStability = False       # ADF must pass in both halves of the formation window
useHalfLife  = False       # keep only pairs whose spread reverts at a usable speed
useZStop     = False       # bail out when the spread blows past zStopLevel
useVolFilter = False       # block new entries when the spread is moving abnormally

recheckMode = "exit"       # "exit" force-closes open trades when the 2-month recheck fails, "blockEntry" only stops new ones, "off" disables the recheck entirely

halfLifeMin  = 3.0
halfLifeMax  = 25.0
zStopLevel   = 4.0
volThreshold = 1.5
volWindow    = 20

halfPValueCutoff = 0.10    # looser than the full-window 0.05 because each half has half the data