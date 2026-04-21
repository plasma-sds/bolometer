import aug_sfutils as sf
import pandas as pd

shot = 40673

diagnostic = "XVU"

amplifications = [
    "VersS6L0", "VersS6L1", "VersS6L2",
    "VersS7L0", "VersS7L1", "VersS7L2"
]

signal = sf.SFREAD(shot, diagnostic, experiment="AUGD")

rows = []

for amp_i, amp in enumerate(amplifications):

    device = signal.getobject(amp)
    sub = device.__dict__["data"]

    stages = [
        "Stufe1_U", "Stufe2_U", "Stufe3_U",
        "Stufe1_O", "Stufe2_O", "Stufe3_O"
    ]

    for stage_i, stage in enumerate(stages):

        data = sub[stage].data.tolist()

        row = ["", "", stage] + data

        if stage_i == 0:
            row[1] = amp

        if amp_i == 0 and stage_i == 0:
            row[0] = f"{shot} - " + diagnostic

        rows.append(row)

df = pd.DataFrame(rows)

df.to_excel(f"xvu_amplification_{shot}.xlsx", index=False, header=False)