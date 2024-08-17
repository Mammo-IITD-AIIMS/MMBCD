import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F

def apply_softmax_and_combine(original_array):
    # Extract the first two columns for softmax
    logits = original_array[:, :2]

    # Apply softmax along the second axis (axis=1)
    softmax_result = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_result /= np.sum(softmax_result, axis=1, keepdims=True)

    # Create a new array with softmax applied to the first two columns and original third column
    new_array = np.concatenate((softmax_result, original_array[:, 2:]), axis=1)

    return new_array

def make_plot(data1, data2, ax_lims, fpi_data=None, ax=None, title = "FND+OURS", data2_label = "Postive BOXES", data1_label = "Negative BOXES"):
    x1,x2,y = ax_lims 
    # Create density plot
    # import pdb; pdb.set_trace()
    sns.kdeplot(data2, label=data2_label, shade=True, ax=ax)
    sns.kdeplot(data1, label=data1_label, shade=True, ax=ax)

    if(fpi_data):
        thresh_data = fpi_data
        cmap = cm.get_cmap('coolwarm')  
        for i,thresh in enumerate(thresh_data):
            color = cmap(i / (len(thresh_data) - 1))
            print(f'made line for {thresh}')
            ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1)
            # ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1, label=f'fpi={fpi[i]}')
            # ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1)
            # ax.text(fpi[i], 3, f'fpi={fpi[i]}', rotation=90, color=color, ha='center', va='center')

    ax.set_xlim(x1, x2)
    ax.set_ylim(0, y)
    # Set plot labels and title
    # ax.text(0.5, -0.9,'{} Proposals'.format(title), ha='center', va='center', fontsize=12)
    # ax.text(0.5, 1.1, '{} Proposals'.format(title), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    # ax.set_title('{} Proposals'.format(title), loc='center', pad=-20)
    ax.set_title(title, rotation='vertical', fontsize=25, weight='bold', x=-0.17, y=0.5, va = 'center')
    ax.set_xlabel('CONFIDENCE', fontsize=12, fontweight ='bold')
    ax.set_ylabel('PDF', fontsize=12, fontweight ='bold')
    

    # Show legend
    legend = ax.legend(prop={'weight': 'bold', 'size': '20'})
    legend.get_title().set_fontweight('bold')
    plt.tight_layout()
    # Show the plot
    # ax.savefig("den_plots/{}_{}_plot.png".format(data,key))
    # ax.clf()

def recall2FPR(logits, labels, fpr_value = 0.3):
    neg_idx = np.where(np.array(labels)==0)[0]
    neg_logits  = sorted([logits[idx] for idx in neg_idx], reverse=True)
    thresh =  neg_logits[int(fpr_value*len(neg_logits))]

    return thresh

def r2f(neg_logs, fpr_value = 0.3):
    neg_logits  = sorted(neg_logs, reverse=True)
    thresh =  neg_logits[int(fpr_value*len(neg_logits))]

    return thresh

def make_data(base_url):
    logits_labels = np.load(base_url + 'logits_labels.npy')
    logits_labels = apply_softmax_and_combine(logits_labels)
    print(logits_labels.shape)
    
    logits_labels = np.array(logits_labels)

    labels = []
    one_logit = []
    for logit in logits_labels: 
        labels.append(int(logit[2]))
        one_logit.append(logit[1])
    
    probs_pos = [one_logit[index] for index, label in enumerate(labels) if label==1]
    probs_neg = [one_logit[index] for index, label in enumerate(labels) if label==0]
    
    thresh_values = []
    vals = [0.025, 0.05, 0.1, 0.3]
    for values in vals:
        thresh_values.append(r2f(probs_neg, fpr_value = values))

    return probs_pos, probs_neg, thresh_values

if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(9.5, 15))
    plt.rcParams['font.size'] = '18'

    base_url = "./full_img_model/"
    probs_pos, probs_neg, thresh_values = make_data(base_url)
    make_plot(probs_pos, probs_neg, (0,1,30), fpi_data=thresh_values, ax=ax[0], title = "Full Image Model", data1_label = "MAL", data2_label = "BEN")

    base_url = "./roi_vision/"
    probs_pos, probs_neg, thresh_values = make_data(base_url)
    make_plot(probs_pos, probs_neg, (0,1,30), fpi_data=thresh_values, ax=ax[1], title = "ROI Vision Model", data1_label = "MAL", data2_label = "BEN")

    base_url = "./ours/"
    probs_pos, probs_neg, thresh_values = make_data(base_url)
    make_plot(probs_pos, probs_neg, (0,1,30), fpi_data=thresh_values, ax=ax[2], title = "Our Model", data1_label = "MAL", data2_label = "BEN")

    plt.savefig('allcomb_final.png')
    

