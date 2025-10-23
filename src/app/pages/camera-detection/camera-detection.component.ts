// camera-detection.component.ts
import { CommonModule } from '@angular/common';
import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { InferenceSession, Tensor, env } from 'onnxruntime-web';

interface Detection {
  class: string;
  confidence: number;
  bbox: number[];
}

@Component({
  selector: 'app-camera-detection',
  imports: [CommonModule],
  templateUrl: './camera-detection.component.html',
  styleUrl: './camera-detection.component.css'
})
export class CameraDetectionComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement') canvasElement!: ElementRef<HTMLCanvasElement>;
  
  // حالة التطبيق
  isCameraOn = false;
  isModelLoaded = false;
  isLoading = false;
  isDarkMode = false;
  lastDetectedObject = '';
  cameraFacingMode: 'user' | 'environment' = 'environment';
  
  // المتغيرات التقنية
  private stream: MediaStream | null = null;
  private session: InferenceSession | null = null;
  private detectionInterval: any = null;
  private lastSpokenObject = '';
  private lastSpokenTime = 0;
  private readonly SPEECH_COOLDOWN = 3000; // 3 ثواني
  
  // أسماء الفئات الخاصة بـ YOLOv8n (مثال)
  private classNames = [
    'شخص', 'دراجة', 'سيارة', 'دراجة نارية', 'طائرة', 'حافلة', 'قطار', 'شاحنة', 'قارب',
    'إشارة مرور', 'صنبور إطفاء', 'علامة توقف', 'عداد ركن', 'مقعد', 'طائر', 'قطة', 'كلب',
    'حصان', 'خروف', 'بقرة', 'فيل', 'دب', 'حمار وحشي', 'زرافة', 'حقيبة ظهر', 'مظلة',
    'حقيبة يد', 'ربطة عنق', 'حقيبة سفر', 'الطائر الطائر', 'القرص الطائر', 'لوح تزلج',
    'لوح تزلج على الماء', 'كرة قدم', 'طائرة ورقية', 'مضرب بيسبول', 'قفاز بيسبول',
    'مزلجة', 'لوح تزلج على الثلج', 'كرة طائرة', 'مضرب تنس', 'زجاجة', 'كأس نبيذ',
    'كأس', 'شوكة', 'سكين', 'ملعقة', 'وعاء', 'موزة', 'تفاح', 'ساندويتش', 'برتقال',
    'بروكلي', 'جزر', 'هوت دوج', 'بيتزا', 'دونات', 'كيك', 'كرسي', 'أريكة', 'نبتة وعاء',
    'سرير', 'طاولة طعام', 'مرحاض', 'تلفاز', 'لابتوب', 'فأرة', 'جهاز تحكم', 'لوحة مفاتيح',
    'هاتف خلوي', 'مايكروويف', 'فرن', 'محمصة خبز', 'حوض', 'ثلاجة', 'كتاب', 'ساعة', 'مزهرية',
    'مقص', 'دب محشي', 'مجفف شعر', 'فرشاة أسنان'
  ];

  // async ngOnInit() {
  //     env.wasm.wasmPaths = 'assets/wasm/';
  //   this.loadThemePreference();
  //   await this.loadModel();
  // }

  ngOnDestroy() {
    this.stopCamera();
    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
    }
  }

  

 async ngOnInit() {
    await this.configureONNXRuntime();
    this.loadThemePreference();
    await this.loadModel();
  }

  private async loadModel() {
    this.isLoading = true;
    try {
      // المحاولة الأولى: استخدام WebGL
      this.session = await InferenceSession.create(
        'assets/models/yolov8n.onnx',
        {
          executionProviders: ['webgl'],
          graphOptimizationLevel: 'all'
        }
      );
      this.isModelLoaded = true;
      console.log('تم تحميل النموذج بنجاح باستخدام WebGL');
    } catch (webglError) {
      console.warn('WebGL غير متاح، جاري استخدام WASM:', webglError);
      
      // المحاولة الثانية: استخدام WASM
      try {
        this.session = await InferenceSession.create(
          'assets/models/yolov8n.onnx',
          {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
          }
        );
        this.isModelLoaded = true;
        console.log('تم تحميل النموذج بنجاح باستخدام WASM');
      } catch (wasmError) {
        console.error('فشل تحميل النموذج مع كلا المزودين:', wasmError);
        this.showModelError();
      }
    } finally {
      this.isLoading = false;
    }
  }

  private async configureONNXRuntime() {
  env.wasm.wasmPaths = '/assets/wasm/';
  env.wasm.numThreads = 1;
  env.wasm.simd = false;
}

private showModelError() {
  alert('تعذر تحميل نموذج التعرف على الأشياء. تأكد من:\n\n1. وجود ملف yolov8n.onnx في مجلد assets/models/\n2. دعم المتصفح لـ WebGL أو WASM\n3. اتصال الإنترنت لتحميل الملفات المطلوبة');
}

  // تبديل وضع الكاميرا
  async toggleCamera() {
    if (this.isCameraOn) {
      this.stopCamera();
    } else {
      await this.startCamera();
    }
  }

  // تبديل بين الكاميرا الأمامية والخلفية
  async switchCamera() {
    if (this.isCameraOn) {
      this.stopCamera();
      this.cameraFacingMode = this.cameraFacingMode === 'user' ? 'environment' : 'user';
      await this.startCamera();
    } else {
      this.cameraFacingMode = this.cameraFacingMode === 'user' ? 'environment' : 'user';
    }
  }

  // تشغيل الكاميرا
  private async startCamera() {
    try {
      this.isLoading = true;
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: this.cameraFacingMode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });

      this.videoElement.nativeElement.srcObject = this.stream;
      this.isCameraOn = true;
      
      // بدء التعرف على الأشياء
      this.startDetection();
      
    } catch (error) {
      console.error('خطأ في تشغيل الكاميرا:', error);
      alert('تعذر الوصول إلى الكاميرا. يرجى التحقق من الأذونات.');
    } finally {
      this.isLoading = false;
    }
  }

  // إيقاف الكاميرا
  private stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.isCameraOn = false;
    this.lastDetectedObject = '';
    
    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
      this.detectionInterval = null;
    }
  }

  // بدء التعرف على الأشياء
  private startDetection() {
    this.detectionInterval = setInterval(async () => {
      if (this.videoElement.nativeElement.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA) {
        await this.detectObjects();
      }
    }, 2000); // كل ثانيتين
  }

  // التعرف على الأشياء
  private async detectObjects() {
    if (!this.session) return;

    try {
      const canvas = this.canvasElement.nativeElement;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // تعيين أبعاد الكانفاس
      canvas.width = 320;
      canvas.height = 320;

      // رسم الفيديو على الكانفاس
      ctx.drawImage(this.videoElement.nativeElement, 0, 0, 320, 320);

      // تحويل الصورة إلى تنسيق مناسب للنموذج
      const imageData = ctx.getImageData(0, 0, 320, 320);
      const input = this.preprocessImage(imageData);

      // تشغيل النموذج
      const results = await this.session.run({ images: input });
      const detections = this.postprocessResults(results);

      // معالجة النتائج
      this.processDetections(detections);

    } catch (error) {
      console.error('خطأ في التعرف على الأشياء:', error);
    }
  }

  // معالجة الصورة قبل إدخالها للنموذج
  private preprocessImage(imageData: ImageData): Tensor {
    const { data, width, height } = imageData;
    const input = new Float32Array(3 * 320 * 320);
    
    for (let i = 0; i < data.length; i += 4) {
      const pixelIndex = i / 4;
      const y = Math.floor(pixelIndex / width);
      const x = pixelIndex % width;
      
      if (x < 320 && y < 320) {
        const targetIndex = y * 320 + x;
        input[targetIndex] = data[i] / 255.0;     // R
        input[320 * 320 + targetIndex] = data[i + 1] / 255.0; // G
        input[2 * 320 * 320 + targetIndex] = data[i + 2] / 255.0; // B
      }
    }
    
    return new Tensor('float32', input, [1, 3, 320, 320]);
  }

  // معالجة نتائج النموذج
  private postprocessResults(results: any): Detection[] {
    const detections: Detection[] = [];
    const output = results.output0.data;
    const outputDims = results.output0.dims;
    
    for (let i = 0; i < outputDims[1]; i++) {
      const confidence = output[i * outputDims[2] + 4];
      
      if (confidence > 0.5) { // ثقة أعلى من 50%
        const classId = output.slice(i * outputDims[2] + 5, (i + 1) * outputDims[2]).indexOf(
          Math.max(...output.slice(i * outputDims[2] + 5, (i + 1) * outputDims[2]))
        );
        
        detections.push({
          class: this.classNames[classId] || `كائن ${classId}`,
          confidence: confidence,
          bbox: [
            output[i * outputDims[2]],     // x
            output[i * outputDims[2] + 1], // y
            output[i * outputDims[2] + 2], // width
            output[i * outputDims[2] + 3]  // height
          ]
        });
      }
    }
    
    return detections;
  }

  // معالجة الاكتشافات
  private processDetections(detections: Detection[]) {
    if (detections.length === 0) {
      this.lastDetectedObject = '';
      return;
    }

    // الحصول على أكثر كائن ثقة
    const bestDetection = detections.reduce((prev, current) => 
      prev.confidence > current.confidence ? prev : current
    );

    this.lastDetectedObject = `${bestDetection.class} (${Math.round(bestDetection.confidence * 100)}%)`;

    // نطق الكائن المكتشف
    this.speakObject(bestDetection.class);
    
    // اهتزاز الجهاز
    if (navigator.vibrate) {
      navigator.vibrate(200);
    }
  }

  // نطق الكائن المكتشف
  private speakObject(objectName: string) {
    const now = Date.now();
    
    // التحقق من عدم تكرار النطق لنفس الكائن في فترة قصيرة
    if (objectName === this.lastSpokenObject && (now - this.lastSpokenTime) < this.SPEECH_COOLDOWN) {
      return;
    }

    if ('speechSynthesis' in window) {
      const msg = new SpeechSynthesisUtterance(`${objectName} أمامك`);
      msg.lang = 'ar-SA';
      msg.rate = 1;
      msg.pitch = 1;
      
      window.speechSynthesis.speak(msg);
      
      this.lastSpokenObject = objectName;
      this.lastSpokenTime = now;
    }
  }

  // إعادة تحميل النموذج
  async reloadModel() {
    this.stopCamera();
    await this.loadModel();
  }

  // تبديل الوضع الليلي
  toggleDarkMode() {
    this.isDarkMode = !this.isDarkMode;
    document.documentElement.classList.toggle('dark', this.isDarkMode);
    localStorage.setItem('darkMode', this.isDarkMode.toString());
  }

  // تحميل تفضيلات الوضع
  private loadThemePreference() {
    const saved = localStorage.getItem('darkMode');
    this.isDarkMode = saved === 'true';
    document.documentElement.classList.toggle('dark', this.isDarkMode);
  }
}