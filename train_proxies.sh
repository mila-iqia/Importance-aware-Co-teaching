echo 'Superconductor'
python3 grad.py --task=Superconductor-RandomForest-v0 --store_path=new_proxies/ > superconductor.txt
echo 'HopperController'
python3 grad.py --task=HopperController-Exact-v0 --store_path=new_proxies/ > hoppercontroller.txt
echo 'AntMorphology'
python3 grad.py --task=AntMorphology-Exact-v0 --store_path=new_proxies/ > antmorphology.txt
echo 'DKittyMorphology'
python3 grad.py --task=DKittyMorphology-Exact-v0 ---store_path=new_proxies/ > dkittymorphology.txt
echo 'TFBind8'
python3 grad.py --task=TFBind8-Exact-v0 --store_path=new_proxies/ > tfbind8.txt
echo 'TFBind10'
python3 grad.py --task=TFBind10-Exact-v0 --store_path=new_proxies/ > tfbind10.txt
echo 'CIFARNAS'
python3 grad.py --task=CIFARNAS-Exact-v0 --store_path=new_proxies/ > nas.txt
