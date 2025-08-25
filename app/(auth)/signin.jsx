import { useRef, useState } from "react";
import {
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import { useRouter } from "expo-router";
import { Formik } from "formik";
import { SafeAreaView } from "react-native-safe-area-context";
import logo from "../../assets/images/NexusLogo.png";
import authSchema from "../../utils/authSchema"; 

const Signin = () => {
  const router = useRouter();

  const passwordInputRef = useRef(null);

  const handleSignin = (values) => {
  };

  return (
    <SafeAreaView className="bg-[#030303] flex-1">
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1, justifyContent: "center" }}
          keyboardShouldPersistTaps="handled"
        >
          <View className="flex justify-center items-center px-6 py-10">
            <View className="flex flex-col items-center mb-12">
              <Image source={logo} style={{ height: 150, width: 150 }} />
              <Text className="text-[#F1EFEC] text-5xl font-extrabold tracking-widest">
                Nexus
              </Text>
              <Text className="text-[#D4C9BE] text-base mt-2">
                Your finances, connected.
              </Text>
            </View>

            <Text className="text-[#F1EFEC] text-3xl font-semibold mb-6">
              Sign In
            </Text>

            <View className="w-full max-w-md">
              <Formik
                initialValues={{
                  email: "",
                  password: "",
                }}
                onSubmit={handleSignin}
                validationSchema={authSchema.pick(['email', 'password'])}
              >
                {({
                  handleChange,
                  handleBlur,
                  values,
                  errors,
                  touched,
                  handleSubmit,
                }) => (
                  <View className="space-y-5">
                    <View>
                      <Text className="text-[#F1EFEC] my-2 font-medium">Email</Text>
                      <TextInput
                        className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                        onChangeText={handleChange("email")}
                        onBlur={handleBlur("email")}
                        value={values.email}
                        keyboardType="email-address"
                        returnKeyType="next"
                        onSubmitEditing={() => {
                          passwordInputRef.current?.focus();
                        }}
                      />
                      {touched.email && errors.email && (
                        <Text className="text-red-500 text-xs mt-1">{errors.email}</Text>
                      )}
                    </View>
                    <View>
                      <Text className="text-[#F1EFEC] my-2 font-medium">Password</Text>
                      <TextInput
                        ref={passwordInputRef}
                        className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                        onChangeText={handleChange("password")}
                        onBlur={handleBlur("password")}
                        value={values.password}
                        secureTextEntry
                        returnKeyType="done"
                        onSubmitEditing={handleSubmit}
                      />
                      {touched.password && errors.password && (
                        <Text className="text-red-500 text-xs mt-1">{errors.password}</Text>
                      )}
                    </View>
                    <TouchableOpacity
                      onPress={handleSubmit}
                      className="bg-[#123458] py-3 rounded-2xl mt-4 shadow-lg shadow-black/50"
                    >
                      <Text className="text-lg text-[#F1EFEC] font-semibold text-center tracking-wide">
                        Sign In
                      </Text>
                    </TouchableOpacity>
                  </View>
                )}
              </Formik>
              <View className="flex-row justify-center mt-7">
                <Text className="text-[#D4C9BE] font-medium">Don&#39;t have an account? </Text>
                <TouchableOpacity onPress={() => router.push('/signup')}>
                    <Text className="text-[#F1EFEC] font-bold">Sign Up</Text>
                </TouchableOpacity>
              </View>

            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

export default Signin;