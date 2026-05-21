#include "kalman_tracker.hpp"

#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise) // Done
    {
        // TODO(student): Implement the constant-velocity Kalman predict step.
        // dt = max(dt, 0)
        // position = position + velocity * dt
        // F = [[1, dt],
        //      [0, 1]]
        // Q = process_noise * [[dt^4 / 4, dt^3 / 2],
        //                      [dt^3 / 2, dt^2]]
        // P = F * P * F^T + Q
        // store the updated position, velocity, and covariance
        dt = std::max(dt, 0.);
        position += velocity * dt;
        double f00 = 1.0;
        double f01 = dt;
        double f10 = 0.0;
        double f11 = 1.0;
        double q00 = process_noise * dt * dt * dt * dt / 4.;
        double q01 = process_noise * dt * dt * dt / 2.;
        double q10 = process_noise * dt * dt * dt / 2.;
        double q11 = process_noise * dt * dt;
        double new_p00 = q00 + p00 + p10 + p01 * dt + p11 * dt * dt;
        double new_p01 = q01 + p01 + p11 * dt;
        double new_p10 = q10 + p10 + p11 * dt;
        double new_p11 = q11 + p11;
        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
        // throw std::logic_error("NotImplementedError: KalmanTracker::AxisFilter::predict");
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise) // Done
    {
        // TODO(student): Implement the 1D position measurement update step.
        // residual = measured_position - position
        // H = [1, 0]
        // S = H * P * H^T + measurement_noise
        // if S is not positive:
        //     return without updating
        // K = P * H^T / S
        // position = position + K[0] * residual
        // velocity = velocity + K[1] * residual
        // P = (I - K * H) * P
        double S = p00 + measurement_noise;
        if (S <= 0.) return;
        double K0 = p00 / S;
        double K1 = p10 / S;
        double residual = measured_position - position;
        position += K0 * residual;
        velocity += K1 * residual;
        double new_p00 = (1. - K0) * p00;
        double new_p01 = (1. - K0) * p01;
        double new_p10 = -K1 * p00 + p10;
        double new_p11 = -K1 * p01 + p11;
        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
        // throw std::logic_error("NotImplementedError: KalmanTracker::AxisFilter::update");
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        // TODO(student): Update tracker state from one measured 3D point.
        // if tracker is not initialized:
        //     initialize x, y, z filters with measurement components
        //     set all velocities to zero
        //     mark tracker as active
        //     return current state
        // predict each axis filter using dt
        // update each axis filter with its measured coordinate
        // return position, velocity, and tracking flag
        if (!tracking_)
        {
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_ = true;
            return stateFromFilters();
        }
        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);
        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);
        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        // TODO(student): Predict target state when a detection is missing.
        // if tracker is not active:
        //     return a non-tracking state
        // predict x, y, z filters with dt
        // return predicted position and velocity
        if (!tracking_) return {};
        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);
        return stateFromFilters();
        // throw std::logic_error("NotImplementedError: KalmanTracker::predict");
    }

    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
