from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    
    image = np.load(filename)
    ret = image - np.mean(image, axis=0)
    return ret

def get_covariance(dataset):

    cov_final = np.array([[0 for i in range(len(dataset[0]))] for j in range(len(dataset[0]))])
    dimension_n = len(dataset[0])-1 #dimensions n-1 for summation
    img_count = -1 #n-1
    
    for i in dataset:
        img_count += 1
        cov = np.outer(i, np.transpose(i))
        cov_final = np.add(cov_final, cov)
    
    cov_final = np.divide(cov_final, dimension_n)*(dimension_n/img_count)
    return cov_final

def get_eig(S, m):

    vals, vect = eigh(S, subset_by_index=[(len(S)-1)-(m-1), (len(S)-1)])
    vals = np.diag(vals)

    dim = len(vals)
    for c in range((int)(m/2)):
        temp = vals[c,c]
        vals[c,c] = vals[(dim-c)-1,(dim - c)-1]
        vals[(dim-c)-1,(dim - c)-1] = temp

    dim = len(vect[0])
    for h in range(len(vect)): #height
        for w in range((int)(m/2)): #width
            temp = vect[h,w]
            vect[h,w] = vect[h,(dim-w)-1]
            vect[h,(dim-w)-1] = temp
                
    return vals, vect

def get_eig_perc(S, perc):

    e_vals, e_vects = get_eig(S, len(S))
    store_vals = [0. for i in range(len(S))]
          
    c_bottom = 0
    for c in range(len(e_vals)):
        c_bottom += e_vals[c,c]

    count = 0
    for b in range(len(e_vals)):
        test = np.divide(e_vals[b,b], c_bottom)
        if (test > perc):
            count += 1
            store_vals[b] = e_vals[b,b]
    
    vals_final = np.array([[0. for i in range(count)] for j in range(count)])
    for i in range(count):
        vals_final[i,i] = store_vals[i]

    vects_final = np.array([[0. for i in range(count)] for j in range(len(S))])
    c = count
    while(c > 0):
        for j in range(len(S)):
            vects_final[j,(count-c)] = e_vects[j,(count-c)]
        c -= 1

    return vals_final, vects_final

def project_image(img, U):


    vect_column = []
    projection = []

    count = 0
    while(count < len(U[0])):
        for b in range(len(U)):
            vect_column = np.append(vect_column, U[b,count])
        vect_column = vect_column.flatten()
        img = img.flatten()
        alpha = np.dot(img,np.transpose(vect_column))
        if (len(projection)==0):
            projection = alpha * vect_column
        else:
            temp = alpha * vect_column
            for l in range(len(projection)):
                projection[l] += temp[l]
        vect_column = []
        count += 1

        
    return projection

def display_image(orig, proj):

    original = orig.reshape(32,32)
    original = np.transpose(original)
    projection = proj.reshape(32,32)
    projection = np.transpose(projection)
    fig = plt.figure()
    
    fig.add_subplot(1,2,1)
    x = plt.imshow(original, aspect='equal')
    plt.title("Original")
    fig.colorbar(x)

    fig.add_subplot(1,2,2)
    y = plt.imshow(projection, aspect='equal')
    plt.title("Projection")
    fig.colorbar(y)
    
    
    plt.show()

    return 0

if __name__ == '__main__':

    x = load_and_center_dataset('./YaleB_32x32.npy')
    s = get_covariance(x)
    print(s)
    L, U = get_eig(s, 2)
    print(L)
    p = project_image(x[0], U)
    display_image(x[0], p)

