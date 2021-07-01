from tkinter import Toplevel, Label, Scale, Button, HORIZONTAL, RIGHT
import cv2
import numpy as np


s = 100
MAX_VALUE = 100
class AdjustFrame(Toplevel):


    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.brightness_value = 255
        self.previous_brightness_value = 255
        self.i=0
        self.contrast_value = 127
        self.previous_contrast_value = 127

        self.original_image = self.master.processed_image
        self.copy=self.original_image
        self.copy1=cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HLS)
        self.processing_image = self.master.processed_image
        self.contrast_label = Label(self, text="Contrast")
        self.contrast_scale = Scale(self, from_=0, to_=255, length=250, resolution=1,
                                       orient=HORIZONTAL)
        self.brightness_label = Label(self, text="Brightness")
        self.brightness_scale = Scale(self, from_=0, to_=510, length=250, resolution=1,
                                      orient=HORIZONTAL)
        # self.saturation_label = Label(self, text="Saturation")
        # self.saturation_scale = Scale(self, from_=0, to_=2, length=250, resolution=0.1,
        #                               orient=HORIZONTAL)
        self.clarity_label = Label(self, text="Blur")
        self.clarity_scale = Scale(self, from_=0, to_=2, length=250, resolution=0.1,
                                    orient=HORIZONTAL)
        self.warmth_label = Label(self, text="Warmth")
        self.warmth_scale = Scale(self, from_=0, to_=1, length=250, resolution=0.05,
                                   orient=HORIZONTAL)
        self.cool_label = Label(self, text="Cool")
        self.cool_scale = Scale(self, from_=0, to_=1, length=250, resolution=0.05,
                                  orient=HORIZONTAL)



        self.r_label = Label(self, text="R")
        self.r_scale = Scale(self, from_=-100, to_=100, length=250, resolution=1,
                             orient=HORIZONTAL)
        self.g_label = Label(self, text="G")
        self.g_scale = Scale(self, from_=-100, to_=100, length=250, resolution=1,
                             orient=HORIZONTAL)
        self.b_label = Label(self, text="B")
        self.b_scale = Scale(self, from_=-100, to_=100, length=250, resolution=1,
                             orient=HORIZONTAL)
        self.apply_button = Button(self, text="Apply")
        self.preview_button = Button(self, text="Preview")
        self.cancel_button = Button(self, text="Cancel")

        self.brightness_scale.set(255)
        self.contrast_scale.set(127)
        self.warmth_scale.set(0)
        self.cool_scale.set(0)
        # self.saturation_scale.set(1)
        self.clarity_scale.set(0)

        self.apply_button.bind("<ButtonRelease>", self.apply_button_released)
        self.preview_button.bind("<ButtonRelease>", self.show_button_release)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)

        self.brightness_label.pack()
        self.brightness_scale.pack()
        self.warmth_label.pack()
        self.warmth_scale.pack()
        self.cool_label.pack()
        self.cool_scale.pack()
        self.clarity_label.pack()
        self.clarity_scale.pack()
        self.contrast_label.pack()
        self.contrast_scale.pack()
        self.r_label.pack()
        self.r_scale.pack()
        self.g_label.pack()
        self.g_scale.pack()
        self.b_label.pack()
        self.b_scale.pack()
        self.cancel_button.pack(side=RIGHT)
        self.preview_button.pack(side=RIGHT)
        self.apply_button.pack()

    def apply_button_released(self, event):
        self.show_button_release(self)
        self.master.processed_image = self.processing_image
        self.close()

    def gamma_function(self,channel, gamma):
        invGamma = 1 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")  # creating lookup table
        channel = cv2.LUT(channel, table)
        return channel

    def show_button_release(self, event):
        temp = self.copy.copy
        self.show_image(self.copy)
        self.original_image = self.copy
        b, g, r = cv2.split(self.processing_image)

        for b_value in b:
            cv2.add(b_value, self.b_scale.get(), b_value)
        for g_value in g:
            cv2.add(g_value, self.g_scale.get(), g_value)
        for r_value in r:
            cv2.add(r_value, self.r_scale.get(), r_value)


        self.rgb = cv2.merge((b, g, r))



        brightness = int((self.brightness_scale.get() - 0) * (255 - (-255)) / (510 - 0) + (-255))

        contrast = int((self.contrast_scale.get() - 0) * (127 - (-127)) / (254 - 0) + (-127))

        if brightness != 0:

            if brightness > 0:

                shadow = brightness

                max = 255

            else:

                shadow = 0
                max = 255 + brightness

            al_pha = (max - shadow) / 255
            ga_mma = shadow

            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(self.rgb, al_pha,
                                  self.rgb, 0, ga_mma)


        else:
            cal = self.rgb

        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)

            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(cal, Alpha,
                                  cal, 0, Gamma)


        clar = self.clarity_scale.get()
        print(clar)
        if clar!=0:
            clar = (int)(clar * 10)
            img = cv2.blur(cal, (clar, clar))
            self.processing_image = img
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
        else:
            self.processing_image = cal

        warmth=self.warmth_scale.get()
        warmth/=2
        img = self.processing_image
        img[:, :, 0] = self.gamma_function(img[:, :, 0], 1-warmth)  # down scaling blue channel
        img[:, :, 2] = self.gamma_function(img[:, :, 2], 1+warmth)  # up scaling red channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = self.gamma_function(hsv[:, :, 1], 1+warmth-0.01)  # up scaling saturation channel
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.processing_image = img

        cool = self.cool_scale.get()
        cool /= 2
        img1 = self.processing_image
        img1[:, :, 0] = self.gamma_function(img1[:, :, 0], 1 + cool)  # down scaling blue channel
        img1[:, :, 2] = self.gamma_function(img1[:, :, 2], 1 - cool)  # up scaling red channel
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv1[:, :, 1] = self.gamma_function(hsv1[:, :, 1], 1 - cool+0.01)  # up scaling saturation channel
        img1 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
        self.processing_image = img1

        self.original_image=temp
        # self.processing_image=cv2.addWeighted(self.processing_image,0.3,self.rgb,0.7,0.0)
        # self.processing_image = cv2.addWeighted(self.processing_image, 0.3, img, 0.7, 0.0)
        self.show_image(self.processing_image)
        # self.destroy()

    def cancel_button_released(self, event):
        self.close()

    def show_image(self, img=None):
        self.master.image_viewer.show_image(img=img)

    def close(self):
        self.show_image()
        self.destroy()
