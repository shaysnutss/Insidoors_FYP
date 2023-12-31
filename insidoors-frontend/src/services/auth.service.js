import axios from "axios";

const API_URL = "http://localhost:30000/api/v1/auth/";

class authService {
    login(email, password) {
        return axios
            .post(API_URL + "authenticate", {
                email,
                password
            })
            .then((response) => {
                if (response.data.accessToken) {
                    localStorage.setItem("user", JSON.stringify(response.data));
                }
                return response.data;
            });
    }

    logout = () => {
        localStorage.removeItem("user");
    };

    getCurrentUser = () => {
        return JSON.parse(localStorage.getItem("user"));
    };
}

export default new authService;