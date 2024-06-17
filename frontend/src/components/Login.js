// import { FcGoogle } from "react-icons/fc";
// import React from "react";
// import '../index.css';
// import { useNavigate } from "react-router-dom";

// const Login = () => {
//     const navigate = useNavigate();

//     const handleGoogleLogin = () => {
//         console.log("Google login clicked");
//     }

//     const handleHome = () => {
//         navigate("/menu");
//     }

//     const handleAddress = () => {
//         navigate("/address");
//     }

//     return (
//         <div className="flex items-center justify-center min-h-screen bg-gray-100">
//             <div className="text-center">
//                 <h1 className="text-4xl font-bold mb-4">Optimization Project</h1>
//                 <p className="text-lg mb-8">TK Bunga Matahari</p>
//                 <button
//                     className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow flex items-center justify-center mx-auto"
//                     onClick={() => {
//                         // Handle Google login logic here
//                         handleGoogleLogin();
//                         handleHome();
//                     }}
//                 >
//                     <FcGoogle className="mr-2" />
//                     Login
//                 </button>
//                 <p className="text-sm mt-4">Are you have address? <a href="" onClick={handleAddress}>Click here</a></p>
//             </div>
//         </div>
//     );
// }

// export default Login;

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Modal from "react-modal";

import { FcGoogle } from "react-icons/fc";
import '../index.css';
// import { randomBytes } from 'crypto';

// Modal styles
const customStyles = {
  content: {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)',
  },
};

// Set app element for accessibility reasons
Modal.setAppElement('#root');

const Login = () => {
    const navigate = useNavigate();
    const [modalIsOpen, setModalIsOpen] = useState(false);
    const [address, setAddress] = useState("");


    // const generateCryptoAddress = () => {
    //     return `0x${randomBytes(20).toString('hex')}`;
    // };

    // Simulate fetching address from the backend
    const fetchAddress = async () => {
        // Placeholder for your backend fetch call
        // Example: return fetch('/api/address').then(res => res.json()).then(data => data.address);
        return "fadfjasd987987349873"; // Dummy address
    };

    const handleGoogleLogin = async () => {
        console.log("Google login clicked");
        const newfetchAddress = await fetchAddress();
        setAddress(newfetchAddress);
        setModalIsOpen(true);
    };

    const closeModal = () => {
        setModalIsOpen(false);
    };

    const handleHome = () => {
        navigate("/menu");
    };

    const handleAddress = (event) => {
        event.preventDefault(); // Prevent default link behavior
        navigate("/address");
    };

    const handleLoginAndNavigate = () => {
        handleGoogleLogin();  // Assuming this function is asynchronous
        // handleHome();
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="text-center">
                <h1 className="text-4xl font-bold mb-4">Optimization Project</h1>
                <p className="text-lg mb-8">TK Bunga Matahari</p>
                <button
                    className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow flex items-center justify-center mx-auto"
                    onClick={handleLoginAndNavigate}
                >
                    <FcGoogle className="mr-2" />
                    Login with Google
                </button>
                <p className="text-sm mt-4">
                    Do you have an address? <a href="#" onClick={handleAddress}>Click here</a>
                </p>
            </div>
            <Modal
                isOpen={modalIsOpen}
                onRequestClose={closeModal}
                style={customStyles}
                contentLabel="Address Modal"
            >
                <h2>Your Address</h2>
                <div className="py-2 text-center">{address}</div>
                <p>save your address for the next login</p>
                <button onClick={closeModal} className="mt-4">
                    Close
                </button>
            </Modal>
        </div>
    );
}

export default Login;
