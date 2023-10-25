#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "MPC.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    std::cout << sdata << std::endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          double a = j[1]["steering_angle"];
          
          /**
           * TODO: Calculate steering angle and throttle using MPC.
           * Both are in between [-1, 1].
           */
          VectorXd x_vals(ptsx.size());
          VectorXd y_vals(ptsy.size());
          x_vals.fill(0.0);
          y_vals.fill(0.0);
          for (int i = 0; i < ptsx.size(); i++) {
              double x = ptsx[i] - px;
              double y = ptsy[i] - py;
              x_vals(i) = x * cos(-psi) - y * sin(-psi);
              y_vals(i) = y * cos(-psi) + x * sin(-psi);

          }
          
          auto coeffs = polyfit(x_vals, y_vals, 3);

          const double delay = 100.0 / 1000.0;
          const double x0 = 0;
          const double y0 = 0;
          const double psi0 = 0;
          const double cte0 = coeffs[0];
          const double epsi0 = - atan(coeffs[1]);

          double x_delay = x0 + (v * cos(psi0) * delay);
          double y_delay = y0 + (v * sin(psi0) * delay);
          double psi_delay = psi0 + ((v / mpc.Lf) * delta * delay);
          double v_delay = v + (a * delay);
          double cte_delay = cte0 + (v * sin(psi0) * delay);
          double epsi_delay = epsi0 - (v * atan(coeffs[1]) * delay / mpc.Lf);


          VectorXd state(6);
          state << x_delay, y_delay, psi_delay, v_delay, cte_delay, epsi_delay;
          std::cout << "state: " << state << std::endl;
          auto vars = mpc.Solve(state, coeffs);

          
          double steer_value;
          double throttle_value;
          steer_value = vars[0];
          throttle_value = vars[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the 
          //   steering value back. Otherwise the values will be in between 
          //   [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          // Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          /**
           * TODO: add (x,y) points to list here, points are in reference to 
           *   the vehicle's coordinate system the points in the simulator are 
           *   connected by a Green line
           */

          for (int i = 2; i < vars.size(); i++) {
              if (i % 2 == 0)
                  mpc_x_vals.push_back(vars[i]);
              else
                  mpc_y_vals.push_back(vars[i]);
          }

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          // Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: add (x,y) points to list here, points are in reference to 
           *   the vehicle's coordinate system the points in the simulator are 
           *   connected by a Yellow line
           */


          double poly_inc = 2.5;
          int num_points = 25;
          for (int i = 0; i < num_points; i++) {
              double x = poly_inc * i;
              next_x_vals.push_back(x);
              next_y_vals.push_back(polyeval(coeffs, x));
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          //   the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          //   around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE SUBMITTING.
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}