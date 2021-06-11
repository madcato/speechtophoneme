
# From Baidu ba-dls-deepspeech - https://github.com/baidu-research/ba-dls-deepspeech
# Character map list


char_map_str = """
y 0
ow 1
aa 2
zh 3
nx 4
ix 5
ih 6
ao 7
sh 8
epi 9
pcl 10
n 11
kcl 12
ax-h 13
t 14
w 15
ah 16
en 17
eng 18
r 19
tcl 20
v 21
aw 22
h# 23
m 24
l 25
axr 26
uh 27
eh 28
p 29
ax 30
hv 31
iy 32
ng 33
q 34
em 35
b 36
th 37
bcl 38
z 39
g 40
el 41
dh 42
ch 43
uw 44
ux 45
k 46
f 47
jh 48
ey 49
d 50
gcl 51
er 52
hh 53
pau 54
ay 55
dcl 56
ae 57
oy 58
dx 59
s 60
""" 

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch

def get_number_of_char_classes():
    ## TODO would be better to check with dataset (once cleaned)
    num_classes = len(char_map)+1 ##need +1 for ctc null char +1 pad
    return num_classes
