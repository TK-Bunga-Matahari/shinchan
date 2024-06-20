import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
    apiKey: "AIzaSyDyvuwneC6nIow32tazthIEzACVLkT5-c8",
    authDomain: "deploytest1-426307.firebaseapp.com",
    projectId: "deploytest1-426307",
    storageBucket: "deploytest1-426307.appspot.com",
    messagingSenderId: "663402851893",
    appId: "1:663402851893:web:16a65c33538b0c6664cb7c",
    measurementId: "G-MJ49FVCL6J"
};

// Initialize Firebase app
const app = initializeApp(firebaseConfig);

// Get auth instance from initialized app
export const auth = getAuth(app);