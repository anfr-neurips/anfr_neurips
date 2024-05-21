# merge all json for simulate centralized training, keeping the same training/validation/testing split
out_dir="./datalists"
python3 merge_two_jsons.py --keys "training,validation" --json_1 ${out_dir}/client_cxr14.json --json_2 ${out_dir}/client_cxp_old.json --json_out ${out_dir}/client_All.json
site_IDs="cxp_young padchest"
for i in ${site_IDs}; do
  python3 merge_two_jsons.py --keys "training,validation" --json_1 ${out_dir}/client_${i}.json --json_2 ${out_dir}/client_All.json --json_out ${out_dir}/client_All.json
done
python3 merge_two_jsons.py --keys testing --json_1 ${out_dir}/client_cxr14_test.json --json_2 ${out_dir}/client_cxp_young_test.json --json_out ${out_dir}/client_All_test.json
python3 merge_two_jsons.py --keys testing --json_1 ${out_dir}/client_padchest_test.json --json_2 ${out_dir}/client_All_test.json --json_out ${out_dir}/client_All_test.json