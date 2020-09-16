import logging

import azure.functions as func
import azureml.core as ml
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import ServicePrincipalAuthentication

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # interactive_auth = InteractiveLoginAuthentication(tenant_id="b88f1ff4-e3ab-4adb-83e6-4ea99d41c665")

    sp = ServicePrincipalAuthentication(tenant_id='b88f1ff4-e3ab-4adb-83e6-4ea99d41c665',
                                    service_principal_id='2e90efa1-d53f-45d4-96d8-7adde8a02cdc',
                                    service_principal_password='vltr8qDMzky_7eqnZLLA2JasZ-1Ss0qY_S'
    )
    query = req.params.get('query')

    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            query = req_body.get('query')

    if query == 'run':
        try:
            ws = Workspace.get(name="vrd-ml",
               subscription_id="b9301f45-7da5-41f6-9125-1331de94f262",
               resource_group="vrd-dev-asia",
               auth=sp
               )
            
            compute_name = 'automl-compute'

            if compute_name in ws.compute_targets:
                compute_target = ws.compute_targets[compute_name]
                if compute_target and type(compute_target) is AmlCompute:
                    print('found compute target. just use it. ' + compute_name)
            else:
                print('creating a new compute target...')
                provisioning_config = AmlCompute.provisioning_configuration(vm_size = 'STANDARD_D2_V2',
                                                                            min_nodes = 0, 
                                                                            max_nodes = 4)
                compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
                compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

            dataset = Dataset.get_by_name(ws, name='datasetfunc')

            train_data, test_data = dataset.random_split(percentage=0.8, seed=223)
            label = "ERP"

            automl_config = AutoMLConfig(task = 'regression',
                            compute_target = compute_name,
                            training_data = train_data,
                            label_column_name = label,
                            validation_data = test_data,
                            # n_cross_validations= 3,
                            primary_metric= 'r2_score',
                            enable_early_stopping= True, 
                            experiment_timeout_hours= 0.3,
                            max_concurrent_iterations= 4,
                            max_cores_per_iteration= -1,
                            verbosity= logging.INFO
                            )

            experiment_name = 'expfunc'
            experiment = Experiment(workspace = ws, name = experiment_name)

            run = experiment.submit(automl_config, show_output = True)                
            run

            run.wait_for_completion()
        except ValueError:
            pass
        return func.HttpResponse("AutoML Run Completed")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
