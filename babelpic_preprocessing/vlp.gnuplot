reset

set terminal pdf
set output "log.pdf"
FILES = system("ls -1 log.tsv")

set datafile separator '\t'
set xlabel "Epoch"
set ylabel "Scores"
set yrange [0:1]
set xrange [0:10]
set xtics nomirror
set ytics nomirror 0.1
#set arrow 1 from 0,0.5 to 50,0.5 nohead lt 0
set key font ",5"

set ylabel "Precision"
#set arrow 1 from 0,0.67 to 50,0.67 nohead lt 0

plot "log.tsv" using 1:2 title "Validation" noenhanced with lines,\
	"log.tsv" using 1:5 title "Test" noenhanced with lines,\
	"log.tsv" using 1:8 title "Test Hard" noenhanced with lines,\


set ylabel "Recall"
#set arrow 1 from 0,0.67 to 50,0.67 nohead lt 0

plot "log.tsv" using 1:3 title "Validation" noenhanced with lines,\
	"log.tsv" using 1:6 title "Test" noenhanced with lines,\
	"log.tsv" using 1:9 title "Test Hard" noenhanced with lines,\


set ylabel "F1"
set arrow 1 from 0,0.67 to 50,0.67 nohead lt 0

plot "log.tsv" using 1:4 title "Validation" noenhanced with lines,\
	"log.tsv" using 1:7 title "Test" noenhanced with lines,\
	"log.tsv" using 1:10 title "Test Hard" noenhanced with lines,\

set ylabel "CC Precision"
#set arrow 1 from 0,0.67 to 50,0.67 nohead lt 0

plot "cc_log.tsv" using 1:2 title "Validation" noenhanced with lines,\
	"cc_log.tsv" using 1:5 title "Test" noenhanced with lines,\
	"cc_log.tsv" using 1:8 title "Test Hard" noenhanced with lines,\


set ylabel "CC Recall"
#set arrow 1 from 0,0.67 to 50,0.67 nohead lt 0

plot "cc_log.tsv" using 1:3 title "Validation" noenhanced with lines,\
	"cc_log.tsv" using 1:6 title "Test" noenhanced with lines,\
	"cc_log.tsv" using 1:9 title "Test Hard" noenhanced with lines,\


set ylabel "CC F1"
set arrow 1 from 0,0.67 to 50,0.67 nohead lt 0

plot "cc_log.tsv" using 1:4 title "Validation" noenhanced with lines,\
	"cc_log.tsv" using 1:7 title "Test" noenhanced with lines,\
	"cc_log.tsv" using 1:10 title "Test Hard" noenhanced with lines,\