
functions {
  real partial_log_lik(array[] int n_slice,
  int start_unused, int end_unused,
  array[] int sender_id,
  array[] int receiver_id,
  array[] real past_weight,
  array[] real past_reci,
  array[] real past_trans,
  array[] real past_in_modu,
  array[] real past_aggro,
  array[] real past_trans_agg,
  vector troopID,
  array[] int troop,
  matrix mean_nodes,
  matrix mean_multi_effects,
  int K,
  real intercept,
  real b_weight,
  real b_reci,
  real b_trans,
  real b_in_modu,
  real b_aggro,
  real b_trans_agg,
  array[] int Y,
  int useVI) {
    real lp = 0;
    for (i in 1:size(n_slice)) {
      int n = n_slice[i];
      real mu = intercept
      + (b_weight)*past_weight[n] 
      + (b_aggro)*past_aggro[n] 
      + (b_reci)*past_reci[n] 
      + (b_trans)*past_trans[n] 
      + (b_trans_agg)*past_trans_agg[n] 
      + (b_in_modu)*past_in_modu[n]
      + troopID[troop[n]]
      + mean_nodes[sender_id[n], 1]
      + mean_nodes[receiver_id[n], 2] 
      + dot_product(mean_multi_effects[sender_id[n], 1:K],
      mean_multi_effects[receiver_id[n], (K+1):(2*K)]);
      
      if (useVI == 1) {
        mu = fmin(fmax(mu, -10), 10);
      }
      
      lp += bernoulli_lpmf(Y[n] | Phi(mu));
    }
    return lp;
  }
}


data {
  
  int n_nodes ;
  int N ;
  array[N] int sender_id ;
  array[N] int receiver_id ;
  int K ; 
  array[N] int Y ;
  
  array[N] real past_weight ;
  array[N] real past_reci ;
  array[N] real past_trans ;
  array[N] real past_in_modu ;
  array[N] real past_aggro ;
  array[N] real past_trans_agg ;
  array[N] real past_group_size ;
  
  int scan_N ;
  int troop_N ;
  array[N] int troop ;
  
  int n_dyads ;
  array[N] int dyad_id ;
  array[N] int send_receive ;
  
  int useVI ;
  
}

transformed data {
  array[N] int n_slice_idx;
  for (n in 1:N) n_slice_idx[n] = n;
}



parameters {
  
  real intercept ;
  
  cholesky_factor_corr[2] corr_nodes ; 
  vector<lower=0>[2] sigma_nodes ;
  matrix[2, n_nodes] z_nodes ;
  
  //cholesky_factor_corr[2] corr_dyads; 
  //real<lower=0> sigma_dyads; 
  //matrix[2, n_dyads] z_dyads ; 
  
  cholesky_factor_corr[K*2] corr_multi_effects ;
  vector<lower=0>[K*2] sigma_multi_effects ;
  matrix[K * 2, n_nodes] z_multi_effects ;
  
  //vector[scan_N] scanID;
  //real<lower=0> sigma_scanID ;
  
  vector[troop_N] troopID_raw ; 
  real<lower=0> sigma_troop ;
  
  //vector[N] effect_adj ;
  //real alpha ;
  real b_weight ;
  real b_reci ;
  real b_trans ;
  real b_in_modu ;
  real b_aggro ;
  real b_trans_agg ;
  //real<lower=0> sigma_effects ;
  
}

transformed parameters{
  
  matrix[n_nodes, 2] mean_nodes ;
  //matrix[n_dyads,2] mean_dyads;
  matrix[n_nodes, K*2] mean_multi_effects ;
  vector[troop_N] troopID = sigma_troop * troopID_raw;

  
  mean_nodes = (diag_pre_multiply(sigma_nodes, corr_nodes) * z_nodes )';
  //mean_dyads = (diag_pre_multiply(rep_vector(sigma_dyads, 2), corr_dyads) * z_dyads)';
  mean_multi_effects = (diag_pre_multiply(sigma_multi_effects, corr_multi_effects) * z_multi_effects )'; 
  
}

model {
  
  //array[N] real mu ;
  intercept ~ normal(-2,3) ;
  
  //node terms
  to_vector(z_nodes) ~ normal(0,1) ; 
  corr_nodes ~ lkj_corr_cholesky(5) ;
  sigma_nodes ~ exponential(1) ; //gamma(1,1) ; 
  
  
  //dyad terms
  //to_vector(z_dyads) ~ normal(0,1) ; 
  //corr_dyads ~ lkj_corr_cholesky(5) ;
  //sigma_dyads ~ gamma(1,1) ;
  
  
  //multi-effect terms
  // Pin down three specific latent positions
  // Pin down latent positions for nodes 1, 2, and 3
  // Node 1
  z_multi_effects[1,1] ~ normal(1, 0.01);  // sender x
  z_multi_effects[2,1] ~ normal(1, 0.01);  // sender y
  z_multi_effects[3,1] ~ normal(1, 0.01);  // receiver x
  z_multi_effects[4,1] ~ normal(1, 0.01);  // receiver y
  
  // Node 2
  z_multi_effects[1,2] ~ normal(0, 0.01);
  z_multi_effects[2,2] ~ normal(0, 0.01);
  z_multi_effects[3,2] ~ normal(0, 0.01);
  z_multi_effects[4,2] ~ normal(0, 0.01);
  
  // Node 3
  z_multi_effects[1,3] ~ normal(-1, 0.01);
  z_multi_effects[2,3] ~ normal(-1, 0.01);
  z_multi_effects[3,3] ~ normal(-1, 0.01);
  z_multi_effects[4,3] ~ normal(-1, 0.01);
  
  for (i in 1:(K * 2)) {
    for (j in 1:n_nodes) {
      if (!(j <= 3 && (i >= 1 && i <= 4))) {
        z_multi_effects[i, j] ~ normal(0, 1);
      }
    }
  }
  
  
  
  corr_multi_effects ~ lkj_corr_cholesky(5) ;
  sigma_multi_effects ~ exponential(1) ;//gamma(1,1) ;
  
  troopID_raw ~ normal(0, 1) ;
  //troopID ~ normal(0, sigma_troop) ;
  sigma_troop ~ exponential(10) ;
  
  //Random intercept for scan ID
  //scanID ~ normal(0, sigma_scanID) ;
  //sigma_scanID ~ gamma(1,1) ;
  
  
  //latent effects
  //alpha ~ normal(0,1) ;
  b_weight ~ normal(0,1) ; 
  b_reci~ normal(0,1) ;
  b_trans~ normal(0,1) ;
  b_in_modu~ normal(0,1) ;
  b_aggro ~ normal(0,1) ;
  b_trans_agg ~ normal(0,1) ;
  
  
  target += reduce_sum(
    partial_log_lik,
    n_slice_idx,
    100,  // grainsize, adjust as needed
    sender_id,
    receiver_id,
    past_weight,
    past_reci,
    past_trans,
    past_in_modu,
    past_aggro,
    past_trans_agg,
    troopID,
    troop,
    mean_nodes,
    mean_multi_effects,
    K,
    intercept,
    b_weight,
    b_reci,
    b_trans,
    b_in_modu,
    b_aggro,
    b_trans_agg,
    Y,
    useVI
    );
    
    
    
}

generated quantities {
  
  array[N] real Y_sim ;
  array[N] real p_pred;

  vector[N] log_lik;
  
  for(n in 1:N){
    
    //real p = Phi(  intercept  + effect_adj[n] +  mean_nodes[sender_id[n],1] + mean_nodes[receiver_id[n],2] + mean_dyads[dyad_id[n], send_receive[n]] + mean_multi_effects[sender_id[n],1:K] * (mean_multi_effects[receiver_id[n],(K+1):(K*2) ] )' ) ;
    p_pred[n] = Phi(  intercept  + b_weight*past_weight[n] + b_reci*past_reci[n]+ b_trans*past_trans[n] + b_aggro * past_aggro[n] + b_trans_agg * past_trans_agg[n] + b_in_modu * past_in_modu[n] + troopID[troop[n]]  + mean_nodes[sender_id[n],1] + mean_nodes[receiver_id[n],2] + mean_multi_effects[sender_id[n],1:K] * (mean_multi_effects[receiver_id[n],(K+1):(K*2) ] )'   ) ; 

  
    Y_sim[n] = bernoulli_rng(p_pred[n]) ;
    log_lik[n] = bernoulli_lpmf(Y[n] | p_pred[n]);
    
  }
  
}
