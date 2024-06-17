import { FcGoogle } from "react-icons/fc";
import React from "react";
import '../index.css';
import { useNavigate } from "react-router-dom";

const Login = () => {
    const navigate = useNavigate();

    const handleGoogleLogin = () => {
        console.log("Google login clicked");
    }

    const handleHome = () => {
        navigate("/menu");
    }

    const handleAddress = () => {
        navigate("/address");
    }

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="text-center">
                <h1 className="text-4xl font-bold mb-4">Optimization Project</h1>
                <p className="text-lg mb-8">TK Bunga Matahari</p>
                <button
                    className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow flex items-center justify-center mx-auto"
                    onClick={() => {
                        // Handle Google login logic here
                        handleGoogleLogin();
                        handleHome();
                    }}
                >
                    <FcGoogle className="mr-2" />
                    Login
                </button>
                <p className="text-sm mt-4">Are you have address? <a href="" onClick={handleAddress}>Click here</a></p>
            </div>
        </div>
    );
}

export default Login;