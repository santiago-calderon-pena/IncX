# CPUs limit is roughly 380 (Estimation)
for i in {1..20}
do
    for j in {1,100,200,300,400,500,600,700,800,900,1000}
    do
        sbatch batch.sh $i $j
    done
done
