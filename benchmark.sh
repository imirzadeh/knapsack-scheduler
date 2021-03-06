for ID in $(seq 0 $(cat models/models.txt))
do
    for N in 1 2 3
    do
        sudo likwid-powermeter python3 -m knapsack.main $ID > $ID_out.txt
        cat $ID_out.txt | grep 'PKG' -A 1 | tail -n 1 | cut -d ' ' -f 3 >> $ID.txt
        cat $ID_out.txt | grep "Runtime" | cut -d ' ' -f 2  >> $ID.txt
        rm $ID_out.txt
        sleep 15
    done
    sleep 10
done
python3 -c 'from knapsack import main; main.generate_report()'
#sudo rm *.txt
