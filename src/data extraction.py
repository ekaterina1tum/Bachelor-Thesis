import os, glob
from phase1_export import export_instance

DATA = os.path.join("/Users/ekaterinatkachenko/PycharmProjects/THESIS/data/MSCDPinstances/010")

print(f"{'instance':<16}{'OF':>14}{'F_prime':>14}{'status':>10}")
print('-' * 54)
for txt in sorted(glob.glob(os.path.join(DATA, '*_RC***.txt'))):
    doc = export_instance(
        txt,
        time_limit=3600,
        max_shift=480,
        out_dir=None,
        params={'MIPFocus': 3, 'Threads': 16},     # <-- add this line
    )
    print(f"{doc['instance']:<16}{doc['objective']:>14.2f}{doc['F_prime']:>14.2f}{doc['status']:>10}")