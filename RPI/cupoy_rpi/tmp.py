import time
a = [('Tug', 0.8104438781738281, (57.11275863647461, 68.79114532470703, 105.61766815185547, 105.97710418701172))]
text0 = 'detect: '+a[0][0]+'\t-confidence: '+str(a[0][1])
text1 = '\ntime is: '+time.asctime(time.localtime(time.time()))
text2 = '\ntime is: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(text0+text2)
