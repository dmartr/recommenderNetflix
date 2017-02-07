"""ALS Matrix Factorization Method."""

def ALS(train, test, user_features, item_features, means, user_bias, item_bias, num_features = 20, lambda_user = 0.1, lambda_item = 0.1):
    stop_thresh = 1e-5
    rmse_list = [float("inf")]
    it = 0 
    max_it = 15
    change = 1
            
    num_user, num_item  = train.shape   
    nz_train, nz_user_itemindices, nz_item_userindices = build_index_groups(train)
    nz_test, niut, nuit = build_index_groups(test)
    
    for i in range(len(nz_item_userindices)):
        nz_item_userindices[i] = nz_item_userindices[i][1]
    for i in range(len(nz_user_itemindices)):
        nz_user_itemindices[i] = nz_user_itemindices[i][1]

    nnz_items_per_user = np.zeros(num_user)
    
    for i in range(num_user):
        nnz_items_per_user[i] = len(nz_user_itemindices[i])
        
    nnz_users_per_item = np.zeros(num_item)
    
    for j in range(num_item):
        nnz_users_per_item[j] = len(nz_item_userindices[j])
    
    lambda_user_diag = np.identity(num_features)
    np.fill_diagonal(lambda_user_diag, lambda_user)

    lambda_item_diag = np.identity(num_features)
    np.fill_diagonal(lambda_item_diag, lambda_item)
    
    #print("Learning with ALS...")
    while ((change > stop_thresh) & (it<max_it)):
        it = it+1 
               
        user_features = np.linalg.inv(item_features.dot(item_features.T) + lambda_user_diag).dot(np.dot(item_features,train.T)).T        
        item_features = np.linalg.inv(user_features.T.dot(user_features) + lambda_item_diag).dot(user_features.T.dot(train))

        prediction = np.dot(user_features, item_features)
            
        rmse = compute_rmse(train, prediction, nonzero_rows_train, nonzero_cols_train )
        print("RMSE train : ", rmse)

        # append it to the list of errors
        rmse_list.append(rmse)
        change = rmse_list[-2]-rmse_list[-1]
    prediction = np.dot(user_features, item_features)  
    prediction = denormalize(prediction, means)

    rmse = compute_rmse(test, prediction, nonzero_rows_test, nonzero_cols_test)
    print("RMSE on test:", rmse)
    return prediction