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

//=========================================================//

// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import Modal from "react-modal";

// import { FcGoogle } from "react-icons/fc";
// import '../index.css';
// // import { randomBytes } from 'crypto';

// // Modal styles
// const customStyles = {
//   content: {
//     top: '50%',
//     left: '50%',
//     right: 'auto',
//     bottom: 'auto',
//     marginRight: '-50%',
//     transform: 'translate(-50%, -50%)',
//   },
// };

// // Set app element for accessibility reasons
// Modal.setAppElement('#root');

// const Login = () => {
//     const navigate = useNavigate();
//     const [modalIsOpen, setModalIsOpen] = useState(false);
//     const [address, setAddress] = useState("");


//     // const generateCryptoAddress = () => {
//     //     return `0x${randomBytes(20).toString('hex')}`;
//     // };

//     // Simulate fetching address from the backend
//     const fetchAddress = async () => {
//         // Placeholder for your backend fetch call
//         // Example: return fetch('/api/address').then(res => res.json()).then(data => data.address);
//         return "fadfjasd987987349873"; // Dummy address
//     };

//     const handleGoogleLogin = async () => {
//         console.log("Google login clicked");
//         const newfetchAddress = await fetchAddress();
//         setAddress(newfetchAddress);
//         setModalIsOpen(true);
//     };

//     const closeModal = () => {
//         setModalIsOpen(false);
//         navigate("/menu");
//     };

//     const handleHome = () => {
//         navigate("/menu");
//     };

//     const handleAddress = (event) => {
//         event.preventDefault(); // Prevent default link behavior
//         navigate("/address");
//     };

//     const handleLoginAndNavigate = () => {
//         handleGoogleLogin();  // Assuming this function is asynchronous
//         // handleHome();
//     };

//     return (
//         <div className="flex items-center justify-center min-h-screen bg-gray-100">
//             <div className="text-center">
//                 <h1 className="text-4xl font-bold mb-4">Optimization Project</h1>
//                 <p className="text-lg mb-8">TK Bunga Matahari</p>
//                 <button
//                     className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow flex items-center justify-center mx-auto"
//                     onClick={handleLoginAndNavigate}
//                 >
//                     <FcGoogle className="mr-2" />
//                     Login with Google
//                 </button>
//                 <p className="text-sm mt-4">
//                     Do you have an address? <a href="#" onClick={handleAddress}>Click here</a>
//                 </p>
//             </div>
//             <Modal
//                 isOpen={modalIsOpen}
//                 onRequestClose={closeModal}
//                 style={customStyles}
//                 contentLabel="Address Modal"
//             >
//                 <h2>Your Address</h2>
//                 <div className="py-2 text-center">{address}</div>
//                 <p>save your address for the next login</p>
//                 <button onClick={closeModal} className="mt-4">
//                     Close
//                 </button>
//             </Modal>
//         </div>
//     );
// }

// export default Login;


import React from "react";
import { useNavigate } from "react-router-dom";
import { FcGoogle } from "react-icons/fc";
import '../index.css';
import { auth } from '../firebase-config';
// import firebase from "firebase/app";
import { GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';
import { useEffect } from "react";


const Login = () => {
    const navigate = useNavigate();

    // const googleProvider = new GoogleAuthProvider();

    useEffect(() => {
        const signInWithGoogle = () => {
            const googleProvider = new GoogleAuthProvider();
            googleProvider.setCustomParameters({ 'hd': 'satudata.digital' });

            signInWithPopup(auth, googleProvider)
                .then((res) => {
                    console.log('User signed in:', res.user);
                    navigate('/menu'); // Redirect after successful login
                })
                .catch((error) => {
                    console.error('Error during sign-in:', error);
                });
        };

        const loginButton = document.getElementById('login');

        if (loginButton) {
            loginButton.addEventListener('click', signInWithGoogle);
        }

        return () => {
            // Safely remove the event listener if the element still exists
            if (loginButton) {
                loginButton.removeEventListener('click', signInWithGoogle);
            }
        };
    }, [navigate]); // Only re-run the effect if navigate changes

    const handleLogout = () => {
        signOut(auth)
            .then(() => {
                console.log('User signed out');
                navigate('/'); // Navigate to home or login page
            })
            .catch((error) => {
                console.error('Error during sign-out:', error);
            });
    };

    // const signInWithGoogle = () => {
    //     signInWithPopup(auth, googleProvider)
    //         .then((res) => {
    //             console.log('[Login Success] currentUser:', res.user);
    //             navigate('/menu'); // Redirect after successful login
    //         })
    //         .catch((error) => {
    //         console.log('[Login Failed]', error);
    //     });
    // };

    const handleAddress = (event) => {
        event.preventDefault(); // Prevent default link behavior
        navigate("/address");
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="text-center">
                <h1 className="text-4xl font-bold mb-4">Optimization Project</h1>
                <p className="text-lg mb-8">TK Bunga Matahari</p>
                <button
                    id="login"
                    className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow flex items-center justify-center mx-auto"
                    // onClick={signInWithGoogle}
                >
                    <FcGoogle className="mr-2" />
                    Login
                </button>
                <button
                    onClick={handleLogout}
                    className="bg-red-200 hover:bg-red-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow mt-4"
                >
                    Logout
                </button>
                <p className="text-sm mt-4">Do you have an address? <a href="#" onClick={handleAddress}>Click here</a></p>
            </div>
        </div>
    );
}

export default Login;
