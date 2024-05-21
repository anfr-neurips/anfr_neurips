data_dir="images_master"
out_dir="./datalists"
site_IDs="cxr14 cxp_old cxp_young padchest"
mkdir ${out_dir}
for site in ${site_IDs}; do
  python3 prepare_data_split.py --data_dir ${data_dir} --site_name ${site} --out_path ${out_dir}
done
