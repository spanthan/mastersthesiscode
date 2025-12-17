import argparse
from ddlchain import *
# =============================================================
# Command-line Interface
# =============================================================

def run_consistency_test(pipe, zero_shot_pipe):
    result = pipe.run_one("The user can edit the network settings if they are an IT staff member.")
    print(json.dumps(result, indent=4))
    result = pipe.run_one("IT staff members without a permit cannot alter the network settings.")
    print(json.dumps(result, indent=4))

    result = zero_shot_pipe.run_one("The user can edit the network settings if they are an IT staff member.")
    print(json.dumps(result, indent=4))
    result = zero_shot_pipe.run_one("IT staff members without a permit cannot alter the network settings.")
    print(json.dumps(result, indent=4))

def run_insurance_test(pipe, zero_shot_pipe):
    test_text = '''An insurer must provide coverage for emergency medical treatment.
However, if the policyholder has committed insurance fraud, the insurer is not required to provide coverage.
    '''
    result = pipe.run_one(test_text)
    print(json.dumps(result, indent=4))

    result = zero_shot_pipe.run_one(test_text)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDL → ASP Pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    pipe = FullPipeline(verbose=args.verbose)
    zero_shot_pipe = ZeroShotPipeline(verbose=args.verbose)

    # pipe.run_from_file(args.file)
    
    # run_consistency_test(pipe, zero_shot_pipe)
    run_insurance_test(pipe, zero_shot_pipe)
    