import re

text = """
0.0794 : BEGIN
            aiStufe1(iCh)=0
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
0.1585 : BEGIN
            aiStufe1(iCh)=1
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
0.316 :  BEGIN
            aiStufe1(iCh)=2
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
0.631 : BEGIN
            aiStufe1(iCh)=3
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
1.259  : BEGIN
            aiStufe1(iCh)=4
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
2.512  : BEGIN
            aiStufe1(iCh)=5
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
5.01   : BEGIN
            aiStufe1(iCh)=6
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
10     : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=0
            aiStufe3(iCh)=0
         END
20     : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=1
            aiStufe3(iCh)=0
         END
40     : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=2
            aiStufe3(iCh)=0
         END
80     : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=3
            aiStufe3(iCh)=0
         END
160    : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=3
            aiStufe3(iCh)=1
         END
320     : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=3
            aiStufe3(iCh)=2
         END
640    : BEGIN
            aiStufe1(iCh)=7
            aiStufe2(iCh)=3
            aiStufe3(iCh)=3 
"""

pattern = re.compile(
    r'([\d.]+)\s*:\s*BEGIN.*?aiStufe1\(iCh\)=(\d+).*?aiStufe2\(iCh\)=(\d+).*?aiStufe3\(iCh\)=(\d+)',
    re.S
)

mapping = {}

for amp, s1, s2, s3 in pattern.findall(text):
    mapping[(int(s1), int(s2), int(s3))] = float(amp)

print(mapping)

# query
print(mapping[(2,0,0)])