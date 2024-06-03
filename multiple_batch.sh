# CPUs limit is roughly 380 (Estimation)
for (( i=1; i<=20; i+=1 ))
do
    for (( j=1; j<=2500; j+=500 ))
    do
        sbatch batch.sh $i $j
    done
done

