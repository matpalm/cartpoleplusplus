cat runs/77a/out | grep EVAL | grep -v STEP | cut -f3 -d' ' | nl | sed -es/^\s*/a\\t/ > /tmp/p
cat runs/77b/out | grep EVAL | grep -v STEP | cut -f3 -d' ' | nl | sed -es/^\s*/b\\t/ >> /tmp/p
