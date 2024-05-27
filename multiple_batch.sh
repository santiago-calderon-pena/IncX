# CPUs limit is roughly 380 (Estimation)
for i in {1..20}
do
    for (( j=1; j<=2500; j+=50 ))
    do
        sbatch batch.sh $i $j
    done
done

