import { Component } from '@angular/core';
import { CameraDetectionComponent } from './pages/camera-detection/camera-detection.component';

@Component({
  selector: 'app-root',
  imports: [CameraDetectionComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'isee';
}
