//
//  ViewController.swift
//  StyleTransfer
//
//  Created by Oleg Poyaganov on 02/08/2017.
//  Copyright © 2017 Prisma Labs, Inc. All rights reserved.
//

import UIKit
import MobileCoreServices
import Photos
import CoreML

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    let imageSize = 720
    
    @IBOutlet weak var loadingView: UIView!
    
    @IBOutlet var buttons: [UIButton]!
    
    @IBOutlet weak var imageView: UIImageView!
    
    private var inputImage = UIImage(named: "input")!
    
    private let models = [
        mosaic().model,
        the_scream().model,
        udnie().model,
        candy().model
    ]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        loadingView.alpha = 0
        
        for btn in buttons {
            btn.imageView?.contentMode = .scaleAspectFill
        }
    }
    
    // MARK: - Actions
    
    @IBAction func saveResult(_ sender: Any) {
        guard let image = imageView.image else {
            return
        }
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: image)
        }, completionHandler: nil)
    }
    
    @IBAction func takePhoto(_ sender: Any) {
        let alert = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
        
        let cameraAction = UIAlertAction(title: "Camera", style: .default) { (action) in
            self.showImagePicker(camera: true)
        }
        alert.addAction(cameraAction)
        
        let libraryAction = UIAlertAction(title: "Photo Library", style: .default) { (action) in
            self.showImagePicker(camera: false)
        }
        alert.addAction(libraryAction)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel) { (action) in
        }
        alert.addAction(cancelAction)

        if UIDevice.current.userInterfaceIdiom == .pad {
            alert.popoverPresentationController?.sourceView = self.view
        }

        present(alert, animated: true, completion: nil)
    }
    
    private func showImagePicker(camera: Bool) {
        let imagePicker = UIImagePickerController()
        if camera {
            imagePicker.sourceType = .camera
            imagePicker.showsCameraControls = true
        } else {
            imagePicker.sourceType = .photoLibrary
        }
        
        imagePicker.delegate = self
        
        imagePicker.mediaTypes = [kUTTypeImage as String]
        imagePicker.allowsEditing = true
        
        present(imagePicker, animated: true, completion: nil)
    }
    
    @IBAction func styleButtonTouched(_ sender: UIButton) {
        let image = inputImage.cgImage!
        let model = models[sender.tag]
        
        toggleLoading(show: true)
        
        DispatchQueue.global(qos: .userInteractive).async {
            let stylized = self.stylizeImage(cgImage: image, model: model)
            
            DispatchQueue.main.async {
                self.toggleLoading(show: false)
                self.imageView.image = UIImage(cgImage: stylized)
            }
        }
    }
    
    private func toggleLoading(show: Bool) {
        navigationItem.leftBarButtonItem?.isEnabled = !show
        navigationItem.rightBarButtonItem?.isEnabled = !show
        
        UIView.animate(withDuration: 0.25) { [weak self] in
            if show {
                self?.loadingView.alpha = 0.85
            } else {
                self?.loadingView.alpha = 0
            }
        }
    }
    
    // MARK: - UIImagePickerControllerDelegate
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        let image = info[UIImagePickerControllerEditedImage] as! UIImage
        inputImage = image
        imageView.image = image
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    // MARK: - Processing
    
    private func stylizeImage(cgImage: CGImage, model: MLModel) -> CGImage {
        let input = StyleTransferInput(input: pixelBuffer(cgImage: cgImage, width: imageSize, height: imageSize))
        let outFeatures = try! model.prediction(from: input)
        let output = outFeatures.featureValue(for: "outputImage")!.imageBufferValue!
        CVPixelBufferLockBaseAddress(output, .readOnly)
        let width = CVPixelBufferGetWidth(output)
        let height = CVPixelBufferGetHeight(output)
        let data = CVPixelBufferGetBaseAddress(output)!
        
        let outContext = CGContext(data: data,
                                   width: width,
                                   height: height,
                                   bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(output),
                                   space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageByteOrderInfo.order32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue)!
        let outImage = outContext.makeImage()!
        CVPixelBufferUnlockBaseAddress(output, .readOnly)

        return outImage
    }
    
    private func pixelBuffer(cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer? = nil
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA , nil, &pixelBuffer)
        if status != kCVReturnSuccess {
            fatalError("Cannot create pixel buffer for image")
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags.init(rawValue: 0))
        let data = CVPixelBufferGetBaseAddress(pixelBuffer!)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue)
        let context = CGContext(data: data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: bitmapInfo.rawValue)
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer!
    }
}

