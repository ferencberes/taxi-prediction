from haversine import haversine
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def compute_mean_haversine(df,pred_keys,label_keys):
    """Compute mean Haversine distance for the given keys"""
    lat_pred_key, lng_pred_key = pred_keys
    lat_label_key, lng_label_key = label_keys
    df["HAVERSINE"] = df.apply(lambda x: haversine((x[lat_pred_key],x[lng_pred_key]),(x[lat_label_key],x[lng_label_key])), axis=1)
    return df["HAVERSINE"].mean(), df

def eval_gbt_models(df, lat_model, lng_model, lat_features, lng_features, use_original_trg=True):
    """Evaluate the performance of a latitude and a longitude GBT regressor models. Output the result in mean Haversine distance."""
    prediction_df = pd.DataFrame(df)
    prediction_df["PRED_LAT"] = lat_model.predict(df[lat_features])
    prediction_df["PRED_LNG"] = lng_model.predict(df[lng_features])
    if use_original_trg:
        prediction_df["LABEL_LAT"] = df["DESTINATION_LAT_FULL"]
        prediction_df["LABEL_LNG"] = df["DESTINATION_LNG_FULL"]
    else:
        prediction_df["LABEL_LAT"] = df["DESTINATION_LAT"]
        prediction_df["LABEL_LNG"] = df["DESTINATION_LNG"]
    prediction_df["HAVERSINE"] = prediction_df.apply(lambda x: haversine((x["PRED_LAT"],x["PRED_LNG"]),(x["LABEL_LAT"],x["LABEL_LNG"])), axis=1)
    return compute_mean_haversine(prediction_df,("PRED_LAT","PRED_LNG"),("LABEL_LAT","LABEL_LNG"))

def init_knn_models(df,feat_order,k=1,cell_key="LOC_KEY",verbose=False):
    """Train k-NN models for each location cells. Then return the trained models."""
    knn_models = {}
    cell_locations = {}
    unique_cell_keys = df[cell_key].unique()
    print "Number of unique cell keys: %i" % len(unique_cell_keys)
    idx = 0
    for loc_key in unique_cell_keys:
        cell_df = df[df[cell_key] == loc_key]
        if len(cell_df) < k:
            continue
        cell_arr = cell_df[feat_order].as_matrix()
        cell_locations[loc_key] = cell_arr
        # the last two columns are the exact destination GPS (we emit this from distance calculation!!!)
        knn_models[loc_key] = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(cell_arr[:,:-2])
        #print loc_key, len(cell_df)
        idx += 1
        if verbose:
            if idx % 10 == 0:
                print idx
    return knn_models, cell_locations

def combine_knn_routes(x, knn_models, cell_locations, feat_order, cell_key="LOC_KEY"):
    """Predict specific for a record with k-NN models. x must be a record. It cannot be a matrix!"""
    if ['DESTINATION_LAT_FULL', 'DESTINATION_LNG_FULL'] != feat_order[-2:]:
        raise RuntimeError("Invalid 'feat_order'")
    loc_key = x[cell_key]
    if loc_key in knn_models:
        # the last two columns are the exact destination GPS (we emit this from distance calculation!!!)
        record = np.array(x[feat_order][:-2]).reshape(1, -1)
        distances, indices = knn_models[loc_key].kneighbors(record)
        distances, indices = distances[0], indices[0] # we depend on that x can be only a record
        #print distances, indices
        k_neigh_df = pd.DataFrame(cell_locations[loc_key][indices,-2:], columns=["LAT","LNG"])
        k_centroid = k_neigh_df.mean()
        return (k_centroid["LAT"],k_centroid["LNG"])
    else:
        return None

def predict_with_knn(df, knn_models, cell_locations, feat_order, cell_key="LOC_KEY", mean_pred_keys=("MEAN_PRED_LAT","MEAN_PRED_LNG")):
    """Predict specific destinations with k-NN models."""
    df["TMP_K_CENTROID"] = df.apply(lambda x: combine_knn_routes(x,knn_models,cell_locations,feat_order,cell_key=cell_key),axis=1)
    df["KNN_PRED_LAT"] = df.apply(lambda x: x[mean_pred_keys[0]] if x["TMP_K_CENTROID"] == None else x["TMP_K_CENTROID"][0],axis=1)
    df["KNN_PRED_LNG"] = df.apply(lambda x: x[mean_pred_keys[1]] if x["TMP_K_CENTROID"] == None else x["TMP_K_CENTROID"][1],axis=1)
    df = df.drop(["TMP_K_CENTROID"],axis=1)