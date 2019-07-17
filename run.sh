bash ./train.sh
sleep 35
sudo bash ./benchmark.sh
sleep 30
python3 -m knapsack.optimize
git add ./report.xlsx
git commit -m 'Run experiment'
git pull --rebase
git push origin master
python3 -c 'from knapsack import helpers; helpers.upload_experiment()'
