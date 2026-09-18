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

// written by bastiaan konings schuiling 2008 - 2015
// this work is public domain. the code is undocumented, scruffy, untested, and should generally not be used for anything important.
// i do not offer support, so don't ask. to be used for inspiration :)

#include "main.hpp"

#include "utils.hpp"

#include "systems/graphics/resources/texture.hpp"
#include <boost/algorithm/string.hpp>
#include <cmath>

float GetQuantizedDirectionBias() {
  DO_VALIDATION;
  return GetConfiguration()->GetReal("gameplay_quantizeddirectionbias", _default_QuantizedDirectionBias);
}

void QuantizeDirection(Vector3 &inputDirection, float bias) {
  DO_VALIDATION;

  // digitize input

  Vector3 inputDirectionNorm = inputDirection.GetNormalized(0);

  int directions = GetConfiguration()->GetInt("gameplay_quantizeddirectioncount", 8);

  radian angle = inputDirectionNorm.GetAngle2D();
  angle /= pi * 2.0f;
  angle = std::round(angle * directions);
  angle /= directions;
  angle *= pi * 2.0f;

  inputDirection = (inputDirectionNorm * (1.0 - bias) + (Vector3(1, 0, 0).GetRotated2D(angle) * bias)).GetNormalized(inputDirectionNorm) * inputDirection.GetLength();
}

Vector3 GetProjectedCoord(const Vector3 &pos3D,
                          boost::intrusive_ptr<Camera> camera,
                          bool *visible) {
  DO_VALIDATION;
  Matrix4 rotMat;
  rotMat.ConstructInverse(camera->GetDerivedPosition(), Vector3(1, 1, 1), camera->GetDerivedRotation());
  float fov = camera->GetFOV();

  Vector3 contextSize3D = GetGraphicsSystem()->GetContextSize();
  float aspect = contextSize3D.coords[0] / contextSize3D.coords[1];
  float zNear;
  float zFar;
  camera->GetCapping(zNear, zFar);

  Matrix4 perspMat;
  perspMat.ConstructProjection(fov, aspect, zNear, zFar);

  Matrix4 resMat = perspMat * rotMat;

  float x, y, z, w;
  resMat.MultiplyVec4(pos3D.coords[0], pos3D.coords[1], pos3D.coords[2], 1, x, y, z, w);

  Vector3 result;
  result.coords[0] = x / w;
  result.coords[1] = y / w;

  result.coords[1] = -result.coords[1];

  result += 1.0;
  result *= 0.5;
  result *= 100;

  if (visible) {
    float depth = fabs(w) > EPSILON ? z / w : 2.0f;
    *visible = w > EPSILON && depth >= -1.0f && depth <= 1.0f &&
               result.coords[0] >= 0.0f &&
               result.coords[0] < 100.0f && result.coords[1] >= 0.0f &&
               result.coords[1] < 100.0f;
  }

  return result;
}

LensCalibration GetLensCalibration(CameraType camera) {
  LensCalibration calibration;
  if (camera != CameraType::STATIC_SIDE_0 &&
      camera != CameraType::STATIC_SIDE_1 &&
      camera != CameraType::STATIC_SIDE_2 &&
      camera != CameraType::STATIC_SIDE_3) {
    return calibration;
  }

  // the following parameters are extracted from a sample recording
  // from the Wolfsburg server:
  // "/data/non-www/permanent-data/recordings/WolfsburgServer/4fe34067-e277-447a-af7e-9e565efad535"

  calibration.enabled = true;
  calibration.k1 = -0.31888890f;
  calibration.k2 = 0.09000000f;
  calibration.aspect = 0.5625f;
  if (camera == CameraType::STATIC_SIDE_0) {
    calibration.center_x = 0.49440938f;
    calibration.center_y = 0.50254971f;
    calibration.undistort_scale = 0.70959521f;
  } else if (camera == CameraType::STATIC_SIDE_1) {
    calibration.center_x = 0.49967515f;
    calibration.center_y = 0.50241035f;
    calibration.undistort_scale = 0.71408503f;
  } else if (camera == CameraType::STATIC_SIDE_2) {
    calibration.center_x = 0.50138474f;
    calibration.center_y = 0.50205743f;
    calibration.undistort_scale = 0.71458701f;
  } else {
    calibration.center_x = 0.50848764f;
    calibration.center_y = 0.50017411f;
    calibration.undistort_scale = 0.70555054f;
  }
  return calibration;
}

Vector3 DistortNormalizedCoordinate(const Vector3 &coordinate,
                                    const LensCalibration &calibration) {
  if (!calibration.enabled) return coordinate;

  const float delta_x = coordinate.coords[0] - 0.5f;
  const float delta_y = coordinate.coords[1] - 0.5f;
  const float normalization =
      4.0f / (calibration.aspect * calibration.aspect + 1.0f);
  const float source_radius =
      std::sqrt((delta_x * delta_x * calibration.aspect *
                     calibration.aspect +
                 delta_y * delta_y) *
                normalization);

  Vector3 result = coordinate;
  if (source_radius < 0.000001f) {
    result.coords[0] = calibration.center_x;
    result.coords[1] = calibration.center_y;
    return result;
  }

  // This is the forward counterpart of UndistortLensCoordinate in the
  // postprocess shader. The source pinhole radius was scaled when rendering
  // the wider intermediate image; restore it before applying radial lens
  // distortion.
  const float radius = source_radius / calibration.undistort_scale;
  const float radius2 = radius * radius;
  const float radial_scale =
      1.0f + calibration.k1 * radius2 +
      calibration.k2 * radius2 * radius2;
  const float coordinate_scale =
      radial_scale / calibration.undistort_scale;
  result.coords[0] = calibration.center_x + delta_x * coordinate_scale;
  result.coords[1] = calibration.center_y + delta_y * coordinate_scale;
  return result;
}

int GetVelocityID(e_Velocity velo, bool treatDribbleAsWalk) {
  DO_VALIDATION;
  int id = 0;
  switch (velo) {
    DO_VALIDATION;
    case e_Velocity_Idle:
      id = 0;
      break;
    case e_Velocity_Dribble:
      id = 1;
      break;
    case e_Velocity_Walk:
      id = 2;
      break;
    case e_Velocity_Sprint:
      id = 3;
      break;
    default:
      id = 0;
      break;
  }
  if (treatDribbleAsWalk && id > 1) id--;
  return id;
}

std::map < e_PositionName, std::vector<Stat> > defaultProfiles;

float CalculateStat(float baseStat, float profileStat, float age,
                    e_DevelopmentCurveType developmentCurveType) {
  DO_VALIDATION;


  float idealAge = 27;
  float ageFactor = curve( 1.0f - NormalizedClamp(fabs(age - idealAge), 0, 13) * 0.5f , 1.0f) * 2.0f - 1.0f; // 0 .. 1
  assert(ageFactor >= 0.0f && ageFactor <= 1.0f);
  //ageFactor = clamp(ageFactor, 0.0f, 1.0f);

  // this factor should roughly be around 1.0f for good players at their top age.
  float agedBaseStat = baseStat * (ageFactor * 0.5f + 0.5f) * 1.2f;

  float agedProfileStat = clamp(profileStat * 2.0f * agedBaseStat, 0.01f, 1.0f); // profile stat * 2 because average == 0.5

  return agedProfileStat;
}
