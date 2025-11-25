file=$1

cat $file | sed "s/%//g" | sed "s/ //g" | sed  "s/MiB//g" | sed "s/MHz//g" | sed "s/W//g"
