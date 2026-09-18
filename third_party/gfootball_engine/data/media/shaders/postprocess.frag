// Copyright 2019 Google LLC & Bastiaan Konings
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#version 150

#pragma optimize(on)

uniform sampler2D map_accumulation; // 0
uniform sampler2D map_modifier;     // 1
uniform sampler2D map_depth;        // 2

uniform float contextWidth;
uniform float contextHeight;
uniform float contextX;
uniform float contextY;

uniform vec2 cameraClip;

uniform float fogScale;

uniform vec2 lensDistortion;
uniform vec2 lensCenter;
uniform float lensAspect;
uniform float lensUndistortScale;
uniform int segmentationMode;

out vec4 stdout;

// http://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/

vec3 LinearToSRGB(vec3 color) {
  vec3 lower = color * 12.92;
  vec3 upper = 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055;
  return mix(lower, upper, step(vec3(0.0031308), color));
}

// For all settings: 1.0 = 100% 0.5=50% 1.5 = 150%
vec3 ContrastSaturationBrightness(vec3 color, float brt, float con, float sat) {
  const vec3 LumCoeff = vec3(0.2125, 0.7154, 0.0721);
  // Increase or decrease these values to adjust r, g and b color channels seperately
  vec3 AvgLumin = vec3(0.5, 0.5, 0.5);
  vec3 brtColor = color * brt;
  vec3 intensity = vec3(dot(brtColor, LumCoeff));
  vec3 satColor = mix(intensity, brtColor, sat);
  vec3 conColor = mix(AvgLumin, satColor, con);

  return conColor;
}

vec3 AlternateContrast(vec3 color, float bias) {
  return color * (1.0f - bias) + (-cos(clamp(color, 0.0f, 1.0f) * 3.14159265f) * 0.5f + 0.5f) * bias;
}

vec3 Compress(vec3 color, float startThreshold, float endThreshold) {
  float range = endThreshold - startThreshold;
  float compressedRange = 1.0f - startThreshold;
  for (int c = 0; c < 3; c++) {
    if (color[c] > startThreshold) {
      color[c] = (color[c] - startThreshold) / range; // 0 to 1
      color[c] = pow(color[c], 0.5f);
      color[c] = startThreshold + color[c] * compressedRange;
    }
  }
  return color;
}

vec2 UndistortLensCoordinate(vec2 distorted) {
  vec2 delta = distorted - lensCenter;
  float normalization = 4.0 / (lensAspect * lensAspect + 1.0);
  float distortedRadius =
      sqrt((delta.x * delta.x * lensAspect * lensAspect +
            delta.y * delta.y) * normalization);
  if (distortedRadius < 0.000001) {
    return vec2(0.5);
  }

  // Invert rd = r * (1 + k1*r^2 + k2*r^4), matching camcalib's model.
  float radius = distortedRadius;
  for (int i = 0; i < 12; ++i) {
    float radius2 = radius * radius;
    float radialScale = 1.0 + lensDistortion.x * radius2 +
                        lensDistortion.y * radius2 * radius2;
    radius = distortedRadius / radialScale;
  }

  // The source pinhole render has a wider FOV than the physical lens. This
  // scale converts its coordinates back to the calibrated sensor projection.
  return vec2(0.5) + delta * (radius / distortedRadius) *
                         lensUndistortScale;
}


void main(void) {
  vec2 texCoord = gl_FragCoord.xy;
  texCoord.x -= contextX;
  texCoord.y -= contextY;
  texCoord.x /= contextWidth;
  texCoord.y /= contextHeight;

  if (lensUndistortScale > 0.0) {
    texCoord = UndistortLensCoordinate(texCoord);
  }

  if (segmentationMode != 0) {
    // Labels are categorical values. Match nearest texture sampling explicitly
    // so the warp never introduces IDs between adjacent players/background.
    ivec2 sourceSize = textureSize(map_accumulation, 0);
    ivec2 sourcePixel = ivec2(floor(texCoord * vec2(sourceSize)));
    sourcePixel = clamp(sourcePixel, ivec2(0), sourceSize - ivec2(1));
    stdout = texelFetch(map_accumulation, sourcePixel, 0);
    return;
  }

  vec4 accum = texture(map_accumulation, texCoord);
  vec3 base = accum.rgb;

  vec4 modifier = texture(map_modifier, texCoord);

  // edge blur
  if (modifier.r > 0.0) {
    vec3 smoothPixel = vec3(0);
    smoothPixel += texture(map_accumulation, texCoord + vec2(0, 1 / contextHeight)).xyz;
    smoothPixel += texture(map_accumulation, texCoord + vec2(1 / contextWidth, 0)).xyz;
    smoothPixel += texture(map_accumulation, texCoord + vec2(0, -1 / contextHeight)).xyz;
    smoothPixel += texture(map_accumulation, texCoord + vec2(-1 / contextWidth, 0)).xyz;
    smoothPixel *= 0.25;
    base = base * (1.0 - modifier.r) + smoothPixel * modifier.r;
    //base = base * (1.0 - modifier.r) + vec3(0, 0, 0) * modifier.r * 0.5 + smoothPixel * modifier.r * 0.5; // cartooney effect
    //base = base * (1.0 - modifier.r) + vec3(0, 0, 0) * modifier.r * 0.8 + smoothPixel * modifier.r * 0.2; // cartooney effect
  }


  // SSAO blur

  int SSAO_blurSize = 4;

  float SSAO = 0.0f; // texture(map_modifier, texCoord).g;

  vec2 texelSize = 1.0f / vec2(textureSize(map_modifier, 0));
  vec2 hlim = vec2(float(-SSAO_blurSize) * 0.5f + 0.5f);
  for (int x = 0; x < SSAO_blurSize; ++x) {
    for (int y = 0; y < SSAO_blurSize; ++y) {
      vec2 offset = (hlim + vec2(float(x), float(y))) * texelSize;
      SSAO += texture(map_modifier, texCoord + offset).g;
    }
  }
  SSAO = SSAO / float(SSAO_blurSize * SSAO_blurSize);
  //SSAO = clamp(SSAO * 3.0f - 2.0f, 0.0f, 1.0f);

  //vec3 fragColor = vec3(SSAO);
  vec3 fragColor = base * SSAO;


  // fog

  float depth = texture(map_depth, texCoord).x;

  // convert from non-linear to linear
  float fragDepth = cameraClip.y / (depth - cameraClip.x);

//  vec3 fogColor = vec3(0.84, 0.98, 1.0);
//  vec3 fogColor = vec3(1.0, 0.9, 0.86);
//  vec3 fogColor = vec3(0.85, 0.65, 1.0);
  vec3 fogColor = vec3(0.85, 0.85, 0.9);

  float fogFactor = clamp(fragDepth * 0.01f * (1.0f - fogScale) - 0.16f * fogScale, 0.0f, 0.25f);

  fragColor = fragColor * (1.0f - fogFactor) + fogColor * fogFactor;
  if (depth > 0.999f) fragColor = fogColor; // fill 'background'/sky

  float brightness = 1.0f;
  float contrastBias = 0.3f;//0.1f; // 0 == normal .. 1 == 'fake hdri'
  float saturation = 0.95f * (0.4f + SSAO * 0.6f); // SSAO shadows are less saturated

  fragColor = ContrastSaturationBrightness(fragColor, brightness, 1.0f, saturation);
  fragColor = AlternateContrast(fragColor, contrastBias);
  fragColor = clamp(fragColor, 0.0, 1.0);
  fragColor = LinearToSRGB(fragColor);

  //gl_FragColor = vec4(fragColor, 0);
  //fragColor = vec3(0, 0.5, 1.0);
  stdout = vec4(fragColor, 0);
}
