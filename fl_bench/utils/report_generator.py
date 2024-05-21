# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Union

from ruamel.yaml import YAML

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.widgets.widget import Widget
import pprint


class ReportGenerator(Widget):
    ALLOWED_FILE_EXTENSIONS = [".yaml", ".yml", ".json"]

    def __init__(
        self,
        results_dir: Union[str, Path] = AppConstants.CROSS_VAL_DIR,
        report_path: Union[str, Path] = "cross_val_results_pretty.json",
        metric_type='single',
        metric_name=None,
    ):
        super(ReportGenerator, self).__init__()

        self.results_dir = Path(results_dir)
        self.report_path = Path(report_path)
        self.final_avg = []
        self.pfl_avg = []
        self.order = []
        self.metric_type = metric_type
        self.metric_name = metric_name

        if self.report_path.suffix not in ReportGenerator.ALLOWED_FILE_EXTENSIONS:
            raise ValueError(f"Report file extension must be be .yaml, .yml, or .json, got {self.report_path.suffix}")

        self.val_results = []

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.val_results.clear()
        elif event_type == AppEventType.VALIDATION_RESULT_RECEIVED:
            model_owner = fl_ctx.get_prop(AppConstants.MODEL_OWNER, None)
            data_client = fl_ctx.get_prop(AppConstants.DATA_CLIENT, None)
            val_results = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT, None)

            if not model_owner:
                self.log_error(fl_ctx, "Unknown model owner, validation result will not be saved", fire_event=False)
            if not data_client:
                self.log_error(fl_ctx, "Unknown data client, validation result will not be saved", fire_event=False)
            if val_results:
                try:
                    dxo = from_shareable(val_results)
                    dxo.validate()

                    if dxo.data_kind == DataKind.METRICS:
                        self.val_results.append(
                            {"data_client": data_client, "model_owner": model_owner, "metrics": dxo.data}
                        )
                        if model_owner == "SRV_server":
                            self.order.append(data_client)
                            if self.metric_type == 'specific':
                                self.final_avg.append(dxo.data[self.metric_name])
                            elif self.metric_type == 'single':
                                self.final_avg.append(*dxo.data.values())
                            elif self.metric_type == 'multi':
                                nest = {"data_owner": data_client, "metrics": dxo.data}
                                self.final_avg.append(nest)
                            else:
                                self.log_error(
                                    fl_ctx, f"Unknown metric type {self.metric_type}", fire_event=False
                                )
                        elif model_owner == data_client:
                            if self.metric_type == 'specific':
                                self.pfl_avg.append(dxo.data[self.metric_name])
                            elif self.metric_type == 'single':
                                self.pfl_avg.append(*dxo.data.values())
                            elif self.metric_type == 'multi':
                                nest = {"data_owner": data_client, "metrics": dxo.data}
                                self.pfl_avg.append(nest)
                            else:
                                self.log_error(
                                    fl_ctx, f"Unknown metric type {self.metric_type}", fire_event=False
                                )
                    else:
                        self.log_error(
                            fl_ctx, f"Expected dxo of kind METRICS but got {dxo.data_kind}", fire_event=False
                        )
                except:
                    self.log_exception(fl_ctx, "Exception in handling validation result", fire_event=False)
        elif event_type == EventType.END_RUN:
            ws = fl_ctx.get_engine().get_workspace()
            run_dir = Path(ws.get_run_dir(fl_ctx.get_job_id()))

            output_dir = run_dir / self.results_dir
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            results = {"val_results": self.val_results}

            output_file_path = output_dir / self.report_path
            if self.report_path.suffix == ".json":
                with open(output_file_path, "w") as f:
                    f.write("Cross Validation Results \n")
                    pretty_json_str = pprint.pformat(json.dumps(results), indent=2)
                    f.write(pretty_json_str)
                    f.write("\n")
                    f.write("Order \n")
                    f.write(f'{self.order} \n')
                    f.write("Average Metrics \n")
                    f.write(f'{self.final_avg} \n')
                    f.write("PFL Metrics \n")
                    f.write(f'{self.pfl_avg} \n')
            else:  # ".yaml" or ".yml"
                yaml = YAML()
                with open(output_file_path, "w") as f:
                    yaml.dump(results, f)
