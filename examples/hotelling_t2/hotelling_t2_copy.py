#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from tsquared import HotellingT2

from sample_loader import load_dataset, load_images, as_features

feat_colors = load_dataset();

reference = 4
train = feat_colors[reference]
test = feat_colors[6]

n_train = train.shape[0]
n_test = test.shape[0]

# Inputs.
print("--- Inputs ---\n")

# Fit and print some attributes.
print("\n--- Hotelling's T-squared fitting on the training set---\n")

hotelling = HotellingT2(alpha=0.15)
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

ax[0].scatter(train[:, 0], train[:, 1], c='r', label='Red')
ax[0].scatter(test[:, 0], test[:, 1], c='g', label='Green')
ax[0].scatter(outliers[:, 0], outliers[:, 1], c='b', label='Green')

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
