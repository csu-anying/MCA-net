if (nextflow.version.matches(">=20.07.1")){
    nextflow.enable.dsl=2
}else{
    // Support lower version of nextflow
    nextflow.preview.dsl=2
}

// set project dir
projectDir = workflow.projectDir

params.result = "Predicting......"
params.mode = "predict"


process check_and_pred{
    tag "envcheck and predict"
    errorStrategy 'terminate'

    output:
    path 'result.txt', emit: result

    """
    CUDA_VISIBLE_DEVICES=-1 python ${projectDir}/check_and_pred.py \
        --work_path ${projectDir}
    echo ${params.result} > result.txt
    """

}

// predict
process testModel{
    tag "predict"

    label 'process_high'

    output:
    path 'result.txt', emit: result
    
    """
    CUDA_VISIBLE_DEVICES=-1 python ${projectDir}/test.py \
        --work_path ${projectDir}
    echo ${params.result} > result.txt
    """
}

process printResult{
    tag "printPred"

    label 'process_high'

    input:
    path result
    file x

    output:
    stdout

    """
    if [ -e "$x" ]; then
        cat $x
    else
        echo "Eroor! Errors in model predictions!"
    fi
    """
}

workflow{
    if (params.mode == "val"){
        // 检验模型的可靠性
        testModel()
        printResult(testModel.out.result, file("${projectDir}/result/valid_evaluation.txt")).view{it.trim()}
    } else {
        // 用户测试
        check_and_pred()
        printResult(check_and_pred.out.result, file("${projectDir}/result/user_predict_results.txt")).view{it.trim()}
    }
}