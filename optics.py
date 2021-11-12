from epics import caput
quadrupoles = [
        'SATUN22.MQUA080:K1L-SET',
        'SATMA02.MQUA010:K1L-SET',
        'SATMA02.MQUA020:K1L-SET',
        'SATMA02.MQUA040:K1L-SET',
        'SATMA02.MQUA050:K1L-SET',
        'SATMA02.MQUA070:K1L-SET',
        'SATBD01.MQUA010:K1L-SET',
        'SATBD01.MQUA030:K1L-SET',
        'SATBD01.MQUA050:K1L-SET',
        'SATBD01.MQUA070:K1L-SET',
        'SATBD01.MQUA090:K1L-SET',
        ]


#Matching Quads K1 values:
#@screen for R12=15m, R11=R22=0
QM1=[4.178613916247179,-3.105571097756030,5.599986006176609,-2.868134034737537] # @streaker, beta_x=1.01012 m, @screen beta_x=220m 5.6m^-2 limit , better in y
QM1B=[1.011035788371912,1.478742377078101,4.279999993501157,-3.873096541807198] # @streaker, beta_x=1.01012 m, @screen beta_x=219m 4.28m^-2 limit, !back up, signs are correct! +++-

QM2=[5.246202884528229,-3.958270299291292,4.841574908760317,-2.915051871844490] # @streaker, beta_x=2.50881 m, @screen beta_x=88.5m 5.25m^-2 limit 
QM2B=[4.279998716820764,-1.804821158078722,4.279999994968661,-3.134981443452437] # @streaker, beta_x=2.51078 m, @screen beta_x=88.5m 4.28m^-2 limit, backup

QM3=[4.878485674961142,-3.559646916851266,4.153733961804121,-2.753887141746631] # @streaker, beta_x=5.00939 m, @screen beta_x=44.3m 4.9m^-2 limit
QM3B=[4.199999995280455,-2.569402055950024,4.199999970709370,-3.424039731930264] # @streaker, beta_x=5.01046 m, @screen beta_x=44.3m 4.28m^-2 limit, backup

QM4=[3.553876009541617,-3.055820617590352,4.220150297193038,-3.015944985300997] # @streaker, beta_x=9.99006 m, @screen beta_x=22.2m 4.28m^-2 limit

QM5=[2.501590370488222,-2.685658518431310,4.279999983400799,-2.983814080614478] # @streaker, beta_x=14.9894 m, @screen beta_x=14.8m 4.28m^-2 limit

QM6=[2.774180668006945,-3.680779029072487,4.194464794292636,-2.284100109431831] # @streaker, beta_x=19.9872 m, @screen beta_x=11.1m 4.28m^-2 limit

QM7=[3.042334552493248,-4.199999996750876,3.820497766101923,-1.470299214300066] # @streaker, beta_x=24.9869  m, @screen beta_x=8.9m 4.28m^-2 limit

QM8=[3.050934859398875,-4.199999981837917,3.392281865967324,-0.8587432449134311] # @streaker, beta_x=29.9871   m, @screen beta_x=7.4m 4.28m^-2 limit


optics = {
        '1': {
            'K1L': QM1,
            'betax': 1.01012
            },
        '1B': {
            'K1L': QM1B,
            'betax': 1.01012,
            },
        '2': {
            'K1L': QM2,
            'betax': 2.50881,
            },
        '2B': {
            'K1L': QM2B,
            'betax': 2.51078,
            },
        '3': {
            'K1L': QM3,
            'betax': 5.00939,
            },
        '3B': {
            'K1L': QM3B,
            'betax': 5.01046,
            },
        '4': {
            'K1L': QM4,
            'betax': 9.99006,
            },
        '5': {
            'K1L': QM5,
            'betax': 14.9894,
            },
        '6': {
            'K1L': QM6,
            'betax': 19.9872,
            },
        '7': {
            'K1L': QM7,
            'betax': 24.9869,
            },
        '8': {
            'K1L': QM8,
            'betax': 29.9871,
            },
        }










#Transport Quads K1 values:
QTMA15=[1.103585863863388,-2.627303531903210]
QTBD15=[1.087834034343101,-2.506256739449366,1.693228000031815,-2.557723926080568,0.7984360365715055]

#for QM1-5
QTMA20A=[-0.01834499922465258,-0.4254829576843816]
QTBD20A=[1.113351264646486,-2.197972116354360,1.225692505080959,-3.939426300961390,1.241743467411070]


#for QM6
QTMA20B=[0.6204373026911869,-1.946635645930276]
QTBD20B=[1.272655140340403,-2.211985331507563,1.074129065788922,-3.877923018008127,1.237912224639985]


#for QM7
QTMA20C=[0.7993004991546979,-2.347066715384223]
QTBD20C=[1.363732375309183,-2.559923683483548,0.9406342658957449,-3.100697470352157,1.157365925032256]



#for QM8
QTMA20D=[1.102973328903388,-2.700600242902217]
QTBD20D=[1.241877721216371,-2.195667795345614,1.263151289866165,-3.912249570139790,1.209308988993182]


#for QM1-4
QTMA10A=[-1.148429484933967,-0.7937775930634683]
QTBD10A=[1.181177690534110,-2.457290126850203,1.613801828961291,-2.479983316270420,0.7308980690147209]


#for QM5
QTMA10B=[-2.073199607742802,0.8372193544537587]
QTBD10B=[0.9641356560432376,-2.397709649042056,1.693140511291732,-2.423980358262178,0.7263773330751191]


#for QM6
QTMA10C=[-2.583883686536702,1.657798428197393]
QTBD10C=[0.7596804378366938,-2.278311719891428,1.750755129712383,-2.449808811213372,0.7255270450556310]






L1=0.08
L2=0.3

QM1N= [x * L1 for x in QM1]
QM1BN= [x * L1 for x in QM1B]

QM2N= [x * L1 for x in QM2]
QM2BN= [x * L1 for x in QM2B]

QM3N= [x * L1 for x in QM3]
QM3BN= [x * L1 for x in QM3B]

QM4N= [x * L1 for x in QM4]
QM5N= [x * L1 for x in QM5]
QM6N= [x * L1 for x in QM6]
QM7N= [x * L1 for x in QM7]
QM8N= [x * L1 for x in QM8]


QTMA15N = [x * L1 for x in QTMA15]
QTBD15N = [x * L2 for x in QTBD15]



QTMA20AN = [x * L1 for x in QTMA20A]
QTBD20AN = [x * L2 for x in QTBD20A]

QTMA20BN = [x * L1 for x in QTMA20B]
QTBD20BN = [x * L2 for x in QTBD20B]

QTMA20CN = [x * L1 for x in QTMA20C]
QTBD20CN = [x * L2 for x in QTBD20C]

QTMA20DN = [x * L1 for x in QTMA20D]
QTBD20DN = [x * L2 for x in QTBD20D]




QTMA10AN = [x * L1 for x in QTMA10A]
QTBD10AN = [x * L2 for x in QTBD10A]

QTMA10BN = [x * L1 for x in QTMA10B]
QTBD10BN = [x * L2 for x in QTBD10B]

QTMA10CN = [x * L1 for x in QTMA10C]
QTBD10CN = [x * L2 for x in QTBD10C]

#B for back up no R-> R12=15 m, R10->R12=10m,R20->R12=20m
#Optics 1-5 should work 3-5 fs resolution 



#@streaker, beta_x=1.01012 m
opt1=QM1N+QTMA15N+QTBD15N
opt1B=QM1BN+QTMA15N+QTBD15N
opt1R10=QM1N+QTMA10AN+QTBD10AN
opt1BR10=QM1BN+QTMA10AN+QTBD10AN
opt1R20=QM1N+QTMA20AN+QTBD20AN
opt1BR20=QM1BN+QTMA20AN+QTBD20AN

# @streaker, beta_x=2.50881 m,
opt2=QM2N+QTMA15N+QTBD15N
opt2B=QM2BN+QTMA15N+QTBD15N
opt2R10=QM2N+QTMA10AN+QTBD10AN
opt2BR10=QM2BN+QTMA10AN+QTBD10AN
opt2R20=QM2N+QTMA20AN+QTBD20AN
opt2BR20=QM2BN+QTMA20AN+QTBD20AN

# @streaker, beta_x=5.00939 m,
opt3=QM3N+QTMA15N+QTBD15N
opt3B=QM3BN+QTMA15N+QTBD15N
opt3R10=QM3N+QTMA10AN+QTBD10AN
opt3BR10=QM3BN+QTMA10AN+QTBD10AN
opt3R20=QM3N+QTMA20AN+QTBD20AN
opt3BR20=QM3BN+QTMA20AN+QTBD20AN

# @streaker, beta_x=9.99006 m,
opt4=QM4N+QTMA15N+QTBD15N
opt4R10=QM4N+QTMA10AN+QTBD10AN
opt4R20=QM4N+QTMA20AN+QTBD20AN

# @streaker, beta_x=14.9894 m,
opt5=QM5N+QTMA15N+QTBD15N
opt5R10=QM5N+QTMA10BN+QTBD10BN
opt5R20=QM5N+QTMA20AN+QTBD20AN

# @streaker, beta_x=19.9872 m
opt6=QM6N+QTMA15N+QTBD15N
opt6R10=QM6N+QTMA10CN+QTBD10CN
opt6R20=QM6N+QTMA20BN+QTBD20BN

# @streaker, beta_x=24.9869  m
opt7=QM7N+QTMA15N+QTBD15N
opt7R20=QM7N+QTMA20CN+QTBD20CN

# @streaker, beta_x=29.9871   m
opt8=QM8N+QTMA15N+QTBD15N
opt8R20=QM8N+QTMA20DN+QTBD20DN




print('opt1=',opt1)



#
#pv10 = [L1*x for x in pv10]
#pv20 = [L1*x for x in pv20]
#pv30 = [L1*x for x in pv30]
#pv40 = [L1*x for x in pv40]
#pv50 = [L1*x for x in pv50]
#pv60 = [L1*x for x in pv60]
#pv70 = [L1*x for x in pv70]
#pv80 = [L1*x for x in pv80]
#pv90 = [L1*x for x in pv90]
#
#print(pv10)
#
#fix0=[SATMA02MQ50,SATMA02MQ70]
#fix0= [L2*x for x in fix0]
#
#
#fix=[SATBD01MQ10,SATBD01MQ30,SATBD01MQ50,SATBD01MQ70,SATBD01MQ90]
#
#fix= [L2*x for x in fix]
#print(fix)
#fix=fix0+fix
#print(fix)
#
#
#pv1=pv10+fix
#pv2=pv20+fix
#pv3=pv30+fix
#pv4=pv40+fix
#pv5=pv50+fix
#pv6=pv60+fix
#pv7=pv70+fix
#pv8=pv80+fix
#pv9=pv90+fix
#print(pv9)
#
#
#
#for i in p:
#    print(i)
#    #pv0.append(i)
#    #pv0.append(caget(i))     #use this function to read from the machine
#print(pv0)
#
#
#for i in range(len(p)):
#    print(i)
#    #caput(p[i], pv1[i])  #use this function to write to the machine
#print('Optics 1')
#print(pv1)
#print('Optics 2')
#print(pv5)
#print('Optics 3')
#print(pv7)


def set_optics(optics):
    for q, k1l in zip(quadrupoles, optics):
        print(q, k1l)
        caput(q, k1l)


#caput(pvname, value)
#
#caput('SARBD01-MBND100:I-SET', 0)
