import numpy as np

def generate_uniform_data(num_samples, ranges, seed=None):
    """
    Generate synthetic uniform data based on specified ranges.

    Parameters:
    - num_samples: Number of samples to generate.
    - ranges: List of tuples specifying the range for each feature.
    - seed: Seed for reproducibility.

    Returns:
    - Numpy array containing synthetic data.
    """
    num_features = len(ranges)
    data = np.zeros((num_samples, num_features))

    if seed is not None:
        np.random.seed(seed)

    for i, (min_val, max_val) in enumerate(ranges):
        data[:, i] = np.random.uniform(min_val, max_val, size=num_samples)

    return data

def kappa(beta):
    """
   Calculate kappa based on a given beta (0 - 1) value.

   Parameters:
   - beta: Input value.

   Returns:
   - Calculated kappa value.
   """
    # 25% data from each interval with edges 10^-10 0.1 1 10 10^10
    if beta <= 0.25:
        kappaexp=(beta/0.25)*(-1+10)-10
    elif beta <=0.5:
        kappaexp=((beta-0.25)/0.25)*(0+1)-1
    elif beta <=0.75:
        kappaexp=((beta-0.5)/0.25)*(1-0)
    else:
        kappaexp=((beta-0.75)/0.25)*(10-1)+1
    return 10**kappaexp

def murad(mudelta,CVdelta):
    """
    Calculate mean radius based on mudelta and CVdelta.

    Parameters:
        - mudelta: Input mudelta value.
        - CVdelta: Input CVdelta value.

    Returns:
        - Calculated mean radius.
    """
    mu=np.log(3)+mudelta*np.log(10) #maximum meanradius is 30 and minimum 3
    sigma=(1-np.abs((2*mudelta-1)))*np.log(10)/6*CVdelta #standard dev at extremes is 0
    return  np.exp(mu+0.5*sigma**2)

def sigmarad(mudelta,CVdelta):
    """
    Calculate standard deviation of radius based on mudelta and CVdelta.

    Parameters:
    - mudelta: Input mudelta value.
    - CVdelta: Input CVdelta value.

    Returns:
    - Calculated standard deviation of radius.
    """
    mu=np.log(3)+mudelta*np.log(10)
    sigma=(1-np.abs((2*mudelta-1)))*np.log(10)/6*CVdelta
    return  np.sqrt((np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2))
def mugamma(mualpha,CValpha):
    """
    Calculate mean gamma based on mualpha and CValpha.

    Parameters:
    - mualpha: Input mualpha value.
    - CValpha: Input CValpha value.

    Returns:
    - Calculated mean gamma.
    """
    mu=(2*mualpha-1)*np.log(10)
    sigma=(1-np.abs((2*mualpha-1)))*np.log(10)/3*CValpha
    return  np.exp(mu+0.5*sigma**2)

def sigmagamma(mualpha,CValpha):
    """
   Calculate standard deviation for gamma based on mualpha and CValpha.

   Parameters:
   - mualpha: Input mualpha value.
   - CValpha: Input CValpha value.

   Returns:
   - Calculated standard deviation of gamma.
   """
    mu=(2*mualpha-1)*np.log(10)
    sigma=(1-np.abs((2*mualpha-1)))*np.log(10)/3*CValpha
    return  np.sqrt((np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2))

# Define ranges for each feature
feature_ranges = [
    (0, 1), #mu_delta
    (0, 1), #CV_delta
    (0, 1), #mu_alpha
    (0, 1), #CV_alpha
    (0, 1) #beta
]
# Number of synthetic samples
num_samples = 10000
seed = 65456
# Generate synthetic data
generated_data = generate_uniform_data(num_samples, feature_ranges, seed=seed)
#generated_data = df.values
new_data=generated_data.copy();

# Calculate muR, sigmaR, muG, sigmaG, and Kappa based on generated_data
muR=np.array([murad(x,y) for x,y in generated_data[:,[0,1]]])
sigmaR=np.array([sigmarad(x,y) for x,y in generated_data[:,[0,1]]])
muG=np.array([mugamma(x,y) for x,y in generated_data[:,[2,3]]])
sigmaG=np.array([sigmagamma(x,y) for x,y in generated_data[:,[2,3]]])
Kappa=np.array([kappa(x) for x in generated_data[:,4]])

# Transform muR, sigmaR, muG, sigmaG into log space
mur = np.log(muR**2/np.sqrt(muR**2+sigmaR**2))
variancer = np.log(1+sigmaR**2/muR**2)
mug = np.log(muG**2/np.sqrt(muG**2+sigmaG**2))
varianceg = np.log(1+sigmaG**2/muG**2)

logmaxparticlesize=np.log(75)
# Define inequalities 
inequality1 = (2/3)*(mur+mug)+(2/3)*5*np.sqrt(variancer+varianceg)-logmaxparticlesize
inequality2 = (1/3)*(2*mur-mug)+(1/3)*5*np.sqrt(4*variancer+varianceg)-logmaxparticlesize

# Filter data based on defined conditions
new_data=np.vstack((muR,sigmaR,muG,sigmaG,Kappa)).T
condition = (inequality1<0) & (inequality2<0)
new_data=new_data[condition]
#save the result
output_file = "input.txt"
np.savetxt(output_file, new_data, fmt='%1.4e', delimiter=" ")
