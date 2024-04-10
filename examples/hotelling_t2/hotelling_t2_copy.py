#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.patches as patches
from tsquared import HotellingT2
from prettytable import PrettyTable
import copy
from sample_loader import load_dataset, load_images, as_features

# "/mnt/c/tmp/output"

def draw_ellipse(mean, S, ax):
	# Compute the eigenvalues and eigenvectors of the covariance matrix
	eigvals, eigvecs = np.linalg.eigh(S)

	# Compute the width and height of the ellipse
	width = 2 * np.sqrt(eigvals[1])
	height = 2 * np.sqrt(eigvals[0])

	# Compute the angle of the ellipse
	angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))

	# Create the ellipse
	ellipse = patches.Ellipse(mean, width, height, angle=angle, fill=False,
		edgecolor='b', linewidth=2)

	# Add the ellipse to the plot
	ax.add_patch(ellipse)

base = "/mnt/c/tmp/output_single"
base = "/mnt/c/tmp/b2"
feat_colors, feat_names = load_dataset(base);

#%%
def compare(train, test):
	same_color = np.array_equal(test, train)
	# n_train = train.shape[0]
	# n_test = test.shape[0]

	hotelling = HotellingT2(alpha=0.05)
	hotelling.fit(train)

	# ucl_baseline = 0.1
	# t2_scores = hotelling.score_samples(test)
	# scaled_t2_scores = hotelling.scaled_score_samples(test,
	# 	ucl_baseline=ucl_baseline)

	preds = hotelling.predict(test)

	num_test_color = sum(preds < 0)
	num_train_color = sum(preds > 0)

	if not same_color:
		acc = num_test_color / (num_test_color + num_train_color)
	else:
		acc = num_train_color / (num_test_color + num_train_color)

	return acc

names = copy.deepcopy(feat_names)
names.insert(0, "---")
table = PrettyTable(names)

num_colors = len(feat_colors)
for a in range(num_colors):
	acc = []
	for b in range(num_colors):
		acc_value = compare(train=feat_colors[a], test=feat_colors[b])
		acc.append("{:.2f}".format(acc_value))
		# print(f"Accuracy between {feat_names[a]} and {feat_names[b]}: {acc}")
	acc.insert(0, feat_names[a])
	table.add_row(acc)
print(table)

#%%
cerna = 1
opalit = 3
polar=4

train = feat_colors[polar]
test = feat_colors[opalit]

n_train = train.shape[0]
n_test = test.shape[0]

# Inputs.
print("--- Inputs ---\n")

# Fit and print some attributes.
print("\n--- Hotelling's T-squared fitting on the training set---\n")

hotelling = HotellingT2(alpha=0.05)
hotelling.fit(train)

print(f"Computed mean vector: {hotelling.mean_}")
print(f"Computed covariance matrix:\n{hotelling.cov_}")
print(f"Hotelling's T-squared UCL: {hotelling.ucl(test)}")

# Compute Hotelling's T-squared score for each sample in the test set.
print("\n--- Hotelling's T-squared scores on the test set ---\n")

ucl_baseline = 0.1
t2_scores = hotelling.score_samples(test)
scaled_t2_scores = hotelling.scaled_score_samples(test,
	ucl_baseline=ucl_baseline)

print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
print(f"Scaled Hotelling's T-squared score for each sample:"
	f"\n{scaled_t2_scores}")

# Classify each sample.
print("\n--- Outlier detection ---\n")

preds = hotelling.predict(test)
outliers = test[preds == -1]

print(f"Prediction for each sample:\n{preds}")
print(f"Detected outliers:\n{outliers}")

# Compute Hotelling's T-squared score for the entire test set.
print("\n--- Hotelling's T-squared score on the test set ---\n")

t2_score = hotelling.score(test)
ucl = n_train / (n_train + 1) * hotelling.ucl_indep_

print(f"Hotelling's T-squared score for the entire test set: {t2_score}")
print(f"Do the training set and the test set come from the same "
	f"distribution? {t2_score <= ucl}")

# Plot scaled Hotelling's T-squared scores and the UCL.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

div = 100
# train
ax[0].scatter(train[::div, 0], train[::div, 1], c='r', label='train')
# ax[0].hist2d(train[::div, 0], train[::div, 1], bins=100, cmap='Reds', alpha=0.5)
draw_ellipse(hotelling.mean_, hotelling.cov_, ax[0])

#test
ax[0].scatter(test[::div, 0], test[::div, 1], c='g', label='test', marker='x')
# ax[0].hist2d(test[::div, 0], test[::div, 1], bins=100, cmap='Greens', alpha=0.5)
draw_ellipse(np.mean(test, axis=0), np.cov(test.T, ddof=1), ax[0])

# ax[0].scatter(outliers[::div, 0], outliers[::div, 1], c='b', label='outlier')
ax[0].legend()

# axes limits in point clouds
ax[0].set_xlim(0, 255)
ax[0].set_ylim(0, 255)

ucl_line = ax[1].axhline(y=ucl_baseline, color='r', linestyle='-')
ax[1].scatter(range(scaled_t2_scores.size), scaled_t2_scores)
ax[1].set_title('Scaled Hotelling\'s T2 scores')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('Scaled Hotelling\'s T2 score')
ucl_line.set_label('UCL')
ax[1].legend()

fig.tight_layout()

nmoje = sum(preds > 0)
ncizi = sum(preds < 0)
print("acc: ", ncizi/(nmoje + ncizi))
#%%
images = load_images()

fig1, ax1 = plt.subplots(nrows=len(images), ncols=2, figsize=(112, 112))
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)

def plot_outlier(ax, image, hotelling):
	preds = hotelling.predict(as_features(image))

	zebra = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
	stripe_width = 10
	for y in range(0, image.shape[0]):
		for x in range(y % stripe_width, image.shape[1], stripe_width):
			zebra[y, x, :] = [255, 0, 255]  # Yellow color	
	mask = np.reshape(preds, (image.shape[0], image.shape[1]))

	ax[0].set_xticks([])
	ax[0].set_yticks([])
	ax[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

	out = mask < 0
	image[out, :] = zebra[out, :]
	
	ax[1].set_xticks([])
	ax[1].set_yticks([])
	ax[1].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

for i, image in enumerate(images):
	plot_outlier(ax1[i], image, hotelling)

fig1.tight_layout()

plt.show()

# %%
