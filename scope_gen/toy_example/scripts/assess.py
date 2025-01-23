from scope_gen.toy_example.data.generate_data import generate_data


def assess_method(n_iterations, n_coverage, gen_model, method_func, test_func, data_set_size, epsilon, 
                  distance_func, first_adm_only, k_max, count_adm=True, **kwargs):
    coverages = []
    sizes = []
    first_adms = []
    for i in range(n_iterations):
        # sample data with maximum size
        cal_data = generate_data(gen_model, data_set_size, epsilon=epsilon, distance_func=distance_func, first_adm_only=first_adm_only, k_max=k_max)
        # generate a pipeline
        output = method_func(data=cal_data, count_adm=count_adm, **kwargs)
        if count_adm:
            output, first_adm = output
            first_adms.append(first_adm)
        # test coverage and mean prediction set size
        test_data = generate_data(gen_model, n_coverage, epsilon=epsilon, distance_func=distance_func, k_max=k_max)
        coverage, size = test_func(test_data, output)
        coverages.append(coverage)
        sizes.append(size)

    return coverages, sizes, first_adms
