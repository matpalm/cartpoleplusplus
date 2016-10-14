grep EVAL $* | grep -v EVALSTEP  | perl -plne's/runs\///;s/:EVAL / /;s/\/out//;'  | cut -f1,3,4 -d' '
